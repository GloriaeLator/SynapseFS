/// `sfs commit <file>` — stores a .safetensors checkpoint as a new commit.
///
/// Alignment against the parent commit runs when all three are true: there
/// IS a parent (a root commit has nothing to diff against — ADR 0004: a
/// version control system must be able to store a checkpoint given nothing
/// but the checkpoint), a topology parses (a real config.json, or none at
/// all — see read_topology_bytes's note on why "none at all" and "an empty
/// {} placeholder" must NOT be treated the same by the parser), and the
/// caller didn't pass --no-align. Otherwise every tensor is stored as its
/// own singleton Full group, exactly as this file did before alignment
/// existed — the documented, well-defined fallback, not a degraded mode.
///
/// store::plan_commit_groups is the align <-> codec glue (see its own header
/// for the six things it does); this file's job is just to gather what it
/// needs — a real base ITensorSource for the parent (commit_tensor_source.hpp),
/// a parsed core::Topology, and a completed align::MatchReport — and to keep
/// doing the two things that were never alignment's job: computing the
/// file's sha256 witness and storing the topology object itself.

#include <CLI/CLI.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>

#include <synapsefs/align/topology_parser.hpp>
#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/format/commit.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/format/st_header.hpp>
#include <synapsefs/stio/st_source.hpp>
#include <synapsefs/stio/st_writer.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_planner.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/lockfile.hpp>
#include <synapsefs/store/manifest_store.hpp>
#include <synapsefs/store/refs.hpp>

#include "../commit_tensor_source.hpp"
#include "../exitcode.hpp"
#include "../header_only_source.hpp"

namespace sfs::app::cmd {

namespace {

std::filesystem::path topology_candidate_path(const std::string& topology_path,
                                              const std::filesystem::path& checkpoint_path) {
    return topology_path.empty() ? checkpoint_path.parent_path() / "config.json"
                                 : std::filesystem::path(topology_path);
}

/// Raw bytes of the user's config.json, or an EMPTY vector if none exists.
/// Used for two different things that must NOT share one sentinel value:
///   - storing the Topology object verbatim (store_topology, below), where
///     "{}" has always been the placeholder for "nothing provided", and
///   - driving align::parse_topology, where an empty span means "no
///     config" and takes the documented shape-inference fallback (every
///     tensor gets its own pinned singleton group — always succeeds), while
///     a two-byte "{}" object is NOT empty and hits parse_topology's
///     `cfg.contains("layers")` check, which is a hard TopologyParse error.
///     Reusing the "{}" placeholder for both would turn "no config.json"
///     into a commit failure, which is wrong — see header_only_source.hpp's
///     own note on this exact distinction for checkout/mount.
core::Result<std::vector<std::byte>> read_topology_bytes(
    const std::string& topology_path, const std::filesystem::path& checkpoint_path) {
    auto candidate = topology_candidate_path(topology_path, checkpoint_path);

    std::error_code ec;
    if (!topology_path.empty() && !std::filesystem::exists(candidate, ec)) {
        return SFS_ERR(NoSuchFile, "topology file not found", candidate.string());
    }
    if (!std::filesystem::exists(candidate, ec)) {
        return std::vector<std::byte>{};  // nothing found: empty, deliberately not "{}"
    }

    std::ifstream f(candidate, std::ios::binary);
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<std::byte> bytes(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(bytes.data()), sz);
    return bytes;
}

/// Store the topology object for this commit. Commit::topology is not
/// optional (docs/spec/10 §3) — a repo that has never seen a real topology
/// still needs every commit to satisfy that shape, hence the "{}" fallback.
core::Result<core::Oid> store_topology(store::BlockStore& blocks,
                                       std::span<const std::byte> topology_bytes) {
    if (!topology_bytes.empty()) return blocks.put(core::ObjectKind::Topology, topology_bytes);
    static constexpr char kEmpty[] = "{}";
    return blocks.put(core::ObjectKind::Topology, std::as_bytes(std::span(kEmpty, 2)));
}

int run_commit(const std::string& file, const std::string& message, const std::string& author_opt,
              const std::string& topology_path, const std::vector<std::string>& pin_output,
              const std::vector<std::string>& pin_input, bool lenient_topology, bool no_align) {
    auto paths = core::RepoPaths::discover(std::filesystem::current_path());
    if (!paths) {
        std::cerr << "error: not a synapsefs repository\n";
        return ExitCode::NotARepository;
    }

    std::filesystem::path checkpoint_path(file);
    std::error_code ec;
    if (!std::filesystem::exists(checkpoint_path, ec)) {
        std::cerr << "error: no such file: " << file << "\n";
        return ExitCode::Failure;
    }

    auto cfg = core::RepoConfig::load(paths->root);
    if (!cfg) {
        std::cerr << "error: " << cfg.error().to_string() << "\n";
        return exit_code_for(cfg.error());
    }

    auto lock = store::RepoLock::acquire(paths->lock(), store::LockMode::Exclusive);
    if (!lock) {
        std::cerr << "error: " << lock.error().to_string() << "\n";
        return exit_code_for(lock.error());
    }

    auto blocks = store::BlockStore::open(*paths, *cfg);
    if (!blocks) {
        std::cerr << "error: " << blocks.error().to_string() << "\n";
        return exit_code_for(blocks.error());
    }

    store::RefStore refs(*paths);
    store::CommitStore commits(**blocks, refs);
    store::ManifestStore manifests(**blocks, commits);

    // HEAD must be a symbolic ref to commit onto — detached HEAD needs
    // `checkout -b` first, matching pre-2.23 git semantics named in refs.hpp.
    auto head = refs.read_head();
    if (!head) {
        std::cerr << "error: " << head.error().to_string() << "\n";
        return exit_code_for(head.error());
    }
    if (head->is_detached()) {
        std::cerr << "error: HEAD is detached; create a branch before committing\n";
        return ExitCode::Failure;
    }
    std::string ref_name = *head->symbolic;

    std::optional<core::Oid> parent;
    auto tip = refs.resolve(ref_name);
    if (tip) {
        parent = *tip;
    } else if (tip.error().kind != core::ErrKind::RefNotFound) {
        std::cerr << "error: " << tip.error().to_string() << "\n";
        return exit_code_for(tip.error());
    }
    // RefNotFound: branch has no commits yet — this is the root commit.

    auto source = stio::StSource::open(checkpoint_path);
    if (!source) {
        std::cerr << "error: " << source.error().to_string() << "\n";
        return exit_code_for(source.error());
    }

    stio::Sha256Stream sha;
    sha.update((*source)->header_bytes());

    auto header_oid = (*blocks)->put(core::ObjectKind::Header, (*source)->header_bytes());
    if (!header_oid) {
        std::cerr << "error: " << header_oid.error().to_string() << "\n";
        return exit_code_for(header_oid.error());
    }

    // ---- Topology: parse once, regardless of whether alignment ends up
    // running — a broken --topology should fail the commit even on a root
    // commit, rather than silently storing garbage that surprises the
    // first real diff later. ----
    auto topo_bytes = read_topology_bytes(topology_path, checkpoint_path);
    if (!topo_bytes) {
        std::cerr << "error: " << topo_bytes.error().to_string() << "\n";
        return exit_code_for(topo_bytes.error());
    }

    align::ParseOptions parse_opts;
    parse_opts.pinned_output_tensors = pin_output;
    parse_opts.pinned_input_tensors = pin_input;
    parse_opts.strict = !lenient_topology;

    auto target_topology = align::parse_topology(**source, *topo_bytes, parse_opts);
    if (!target_topology) {
        std::cerr << "error: config.json: " << target_topology.error().to_string() << "\n";
        return exit_code_for(target_topology.error());
    }

    format::Manifest manifest;
    manifest.file.name = checkpoint_path.filename().string();
    manifest.file.header_block = *header_oid;
    manifest.file.total_bytes = (*source)->total_bytes();

    for (const auto& entry : (*source)->buffer_layout()) {
        format::BufferEntry be;
        be.tensor = entry.tensor;
        be.off = entry.off;
        be.nbytes = entry.nbytes;
        be.group = entry.tensor;
        manifest.buffer.push_back(std::move(be));
    }

    const bool use_alignment = !no_align && parent.has_value();
    std::unordered_map<std::string, format::GroupEntry> group_entries;

    if (use_alignment) {
        auto parent_commit = commits.read(*parent);
        if (!parent_commit) {
            std::cerr << "error: " << parent_commit.error().to_string() << "\n";
            return exit_code_for(parent_commit.error());
        }
        auto parent_manifest = manifests.read(parent_commit->manifest);
        if (!parent_manifest) {
            std::cerr << "error: " << parent_manifest.error().to_string() << "\n";
            return exit_code_for(parent_manifest.error());
        }
        auto parent_header_bytes =
            (*blocks)->get(parent_manifest->file.header_block, core::ObjectKind::Header);
        if (!parent_header_bytes) {
            std::cerr << "error: " << parent_header_bytes.error().to_string() << "\n";
            return exit_code_for(parent_header_bytes.error());
        }
        auto parent_header = format::parse_st_header(*parent_header_bytes);
        if (!parent_header) {
            std::cerr << "error: " << parent_header.error().to_string() << "\n";
            return exit_code_for(parent_header.error());
        }

        // Best-effort, same as checkout/mount: absent only for a parent
        // committed before alignment existed, or one whose own config
        // didn't parse — never fatal here. A parent with no topology
        // cannot have stored anything Delta with a secondary dependency
        // (plan_commit_groups requires a topology to ever choose Delta), so
        // a null base topology is always correct on read, never silently
        // wrong.
        auto parent_topology = app::load_commit_topology(**blocks, *parent_commit, *parent_manifest);

        codec::ReadCtx base_ctx;
        base_ctx.blocks = blocks->get();
        base_ctx.manifest = &*parent_manifest;
        base_ctx.history = &manifests;
        base_ctx.max_depth = cfg->max_chain_depth;
        base_ctx.topology = parent_topology ? &*parent_topology : nullptr;

        app::CommitTensorSource base_source(std::move(*parent_header), base_ctx);

        align::Matcher matcher(base_source, **source, *target_topology, align::MatchOptions{});
        auto report = matcher.run();
        if (!report) {
            std::cerr << "error: alignment failed: " << report.error().to_string() << "\n";
            return exit_code_for(report.error());
        }

        // Only tensors the PARENT actually has a group for get a base to
        // diff against; a tensor new to this commit (absent from the
        // parent's manifest) has no entry here and plan_commit_groups
        // correctly falls it through to Full, matching ParentTensorInfo's
        // own documented contract.
        std::unordered_map<std::string, store::ParentTensorInfo> parent_info;
        for (const auto& [tensor_name, ge] : parent_manifest->groups) {
            store::ParentTensorInfo info;
            info.parent_commit = *parent;
            info.chain_depth = ge.chain_depth;
            parent_info.emplace(tensor_name, info);
        }

        auto planned = store::plan_commit_groups(base_source, **source, *target_topology, *report,
                                                  parent_info, **blocks, *cfg, codec::EncodeOptions{});
        if (!planned) {
            std::cerr << "error: " << planned.error().to_string() << "\n";
            return exit_code_for(planned.error());
        }
        group_entries = std::move(*planned);

        // plan_commit_groups already read every tensor once (from `base`
        // for Delta groups, from `target` for Full ones); the file's sha256
        // witness still needs its own pass over TARGET bytes in file order,
        // since that is what reconstruction must reproduce regardless of
        // how any individual tensor ended up stored.
        std::vector<std::byte> tbuf;
        for (const auto& entry : manifest.buffer) {
            tbuf.resize(static_cast<std::size_t>(entry.nbytes));
            auto n = (*source)->read_raw(entry.off, tbuf);
            if (!n) {
                std::cerr << "error: " << n.error().to_string() << "\n";
                return exit_code_for(n.error());
            }
            if (*n != tbuf.size()) {
                std::cerr << "error: short read on tensor " << entry.tensor << "\n";
                return ExitCode::Failure;
            }
            sha.update(tbuf);
        }
    } else {
        // No parent (root commit), or alignment explicitly disabled: every
        // tensor stored Full, verbatim — one read per tensor, sha and the
        // Raw block computed together, exactly as before alignment existed.
        std::vector<std::byte> tbuf;
        for (const auto& entry : manifest.buffer) {
            tbuf.resize(static_cast<std::size_t>(entry.nbytes));
            auto n = (*source)->read_raw(entry.off, tbuf);
            if (!n) {
                std::cerr << "error: " << n.error().to_string() << "\n";
                return exit_code_for(n.error());
            }
            if (*n != tbuf.size()) {
                std::cerr << "error: short read on tensor " << entry.tensor << "\n";
                return ExitCode::Failure;
            }
            sha.update(tbuf);

            auto block_oid = (*blocks)->put(core::ObjectKind::Raw, tbuf);
            if (!block_oid) {
                std::cerr << "error: " << block_oid.error().to_string() << "\n";
                return exit_code_for(block_oid.error());
            }

            format::GroupEntry g;
            g.mode = format::GroupMode::Full;
            g.block = *block_oid;
            g.chain_depth = 0;
            group_entries[entry.tensor] = g;
        }
    }

    for (auto& [name, ge] : group_entries) manifest.groups[name] = std::move(ge);
    manifest.file.sha256 = sha.finish_hex();

    if (auto st = manifest.validate(); !st) {
        std::cerr << "error: built an invalid manifest: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }

    auto manifest_oid = manifests.write(manifest);
    if (!manifest_oid) {
        std::cerr << "error: " << manifest_oid.error().to_string() << "\n";
        return exit_code_for(manifest_oid.error());
    }

    auto topology_oid = store_topology(**blocks, *topo_bytes);
    if (!topology_oid) {
        std::cerr << "error: " << topology_oid.error().to_string() << "\n";
        return exit_code_for(topology_oid.error());
    }

    format::Commit commit;
    if (parent) commit.parents = {*parent};
    commit.manifest = *manifest_oid;
    commit.topology = *topology_oid;
    commit.timestamp = format::now_timestamp();
    const char* env_author = std::getenv("USER");
    commit.author = author_opt.empty() ? (env_author ? env_author : "unknown") : author_opt;
    commit.message = message.empty() ? "(no message)" : message;

    auto commit_oid = commits.commit_and_advance(commit, ref_name, parent);
    if (!commit_oid) {
        std::cerr << "error: " << commit_oid.error().to_string() << "\n";
        return exit_code_for(commit_oid.error());
    }

    std::size_t delta_count = 0;
    for (const auto& [name, ge] : manifest.groups) {
        if (ge.mode == format::GroupMode::Delta) ++delta_count;
    }

    std::cout << "[" << ref_name << " " << commit_oid->abbrev() << "] " << commit.message << "\n";
    std::cout << " " << manifest.buffer.size() << " tensor(s), " << manifest.file.total_bytes
              << " bytes";
    if (use_alignment) {
        std::cout << " (" << delta_count << " stored as deltas against "
                  << parent->abbrev() << ")";
    }
    std::cout << "\n";
    return ExitCode::Ok;
}

}  // namespace

void register_commit(CLI::App& app, int& exit_code) {
    static std::string file;
    static std::string message;
    static std::string author;
    static std::string topology;
    static std::vector<std::string> pin_output;
    static std::vector<std::string> pin_input;
    static bool lenient_topology = false;
    static bool no_align = false;
    auto* c = app.add_subcommand("commit", "Store a .safetensors checkpoint as a new commit");
    c->add_option("file", file, "Path to the .safetensors checkpoint")->required();
    c->add_option("-m,--message", message, "Commit message");
    c->add_option("--author", author, "Author (default: $USER)");
    c->add_option("--topology", topology,
                  "Path to config.json describing the tensor topology "
                  "(default: <file's directory>/config.json if present)");
    c->add_option("--pin-output", pin_output,
                  "Tensor whose output axis must never be permuted (e.g. a classifier "
                  "head); repeatable");
    c->add_option("--pin-input", pin_input,
                  "Tensor whose input axis must never be permuted; repeatable");
    c->add_flag("--lenient-topology", lenient_topology,
                "Tolerate parts of config.json the parser can't model instead of failing "
                "the commit");
    c->add_flag("--no-align", no_align,
                "Store every tensor Full, skipping weight-matching alignment against the "
                "parent commit");
    c->callback([&exit_code] {
        exit_code = run_commit(file, message, author, topology, pin_output, pin_input,
                               lenient_topology, no_align);
    });
}

}  // namespace sfs::app::cmd
