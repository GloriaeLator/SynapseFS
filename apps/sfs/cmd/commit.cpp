/// `sfs commit <file>` — stores a .safetensors checkpoint as a new commit.
///
/// Real alignment now: --topology's config.json is parsed (align::
/// topology_parser), and for a NON-root commit the parent checkpoint is
/// reconstructed (app::ReconstructedTensorSource, streamed through the same
/// codec::read_range every read path uses), matched against the new
/// checkpoint (align::Matcher), and turned into per-tensor Full/Delta
/// decisions (store::plan_commit_groups) — the commit_planner.cpp glue this
/// branch built earlier finally has a caller. A root commit (no parent) has
/// nothing to diff against and stays Full-only, same as before, but now
/// stores a REAL parsed topology when one is available instead of always a
/// placeholder — so the *next* commit has something real to align against.
///
/// Any failure in the alignment/planning path (Matcher, plan_commit_groups,
/// or loading the parent's own topology/data) falls back to the old
/// Full-only behaviour with a warning on stderr, rather than failing the
/// commit outright: losing a dedup opportunity is recoverable, refusing to
/// store a checkpoint at all is not.

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <unordered_map>

#include <synapsefs/align/matcher.hpp>
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

#include "../exitcode.hpp"
#include "../header_only_source.hpp"
#include "../reconstructed_source.hpp"

namespace sfs::app::cmd {

namespace {

/// The bytes to STORE as this commit's topology object: the real config.json
/// if one was given (or found alongside the checkpoint), else the canonical
/// "{}" placeholder — unchanged from before. Kept separate from what gets
/// PARSED (see below): parse_topology treats "{}" as a real-but-empty config
/// and hard-errors ("config.json has no 'layers' array"), whereas a
/// genuinely empty span takes its documented inference path instead. This
/// function's job is only ever "what do we persist", not "what do we feed
/// the parser".
core::Result<std::vector<std::byte>> load_topology_bytes(const std::string& topology_path,
                                                          const std::filesystem::path& checkpoint_path) {
    std::filesystem::path candidate = topology_path.empty()
        ? checkpoint_path.parent_path() / "config.json"
        : std::filesystem::path(topology_path);

    std::error_code ec;
    if (!topology_path.empty() && !std::filesystem::exists(candidate, ec)) {
        return SFS_ERR(NoSuchFile, "topology file not found", candidate.string());
    }

    if (!std::filesystem::exists(candidate, ec)) {
        static constexpr char kEmpty[] = "{}";
        return std::vector<std::byte>(reinterpret_cast<const std::byte*>(kEmpty),
                                      reinterpret_cast<const std::byte*>(kEmpty) + 2);
    }

    std::ifstream f(candidate, std::ios::binary);
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<std::byte> bytes(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(bytes.data()), sz);
    return bytes;
}

/// Reconstruct the parent commit, match it against `target`, and plan
/// per-tensor Full/Delta storage. Isolated into one function so run_commit
/// can fall back to Full-only on ANY failure here with a single call site,
/// rather than threading a fallback through several nested branches.
core::Result<std::unordered_map<std::string, format::GroupEntry>> plan_with_alignment(
    store::BlockStore& blocks, store::CommitStore& commits, store::ManifestStore& manifests,
    const core::Oid& parent, core::ITensorSource& target, const core::Topology& topo,
    const core::RepoConfig& cfg) {
    auto parent_commit = commits.read(parent);
    if (!parent_commit) return std::unexpected(parent_commit.error());

    auto parent_manifest = manifests.read(parent_commit->manifest);
    if (!parent_manifest) return std::unexpected(parent_manifest.error());

    auto parent_header_bytes = blocks.get(parent_manifest->file.header_block, core::ObjectKind::Header);
    if (!parent_header_bytes) return std::unexpected(parent_header_bytes.error());

    auto parent_st_header = format::parse_st_header(*parent_header_bytes);
    if (!parent_st_header) return std::unexpected(parent_st_header.error());

    // Best-effort: absent for a parent stored under the old always-"{}"
    // behaviour, in which case every one of its groups is Full and
    // codec::read_range never needs a topology to reconstruct them anyway.
    auto parent_topo = app::load_commit_topology(blocks, *parent_commit, *parent_manifest);

    codec::ReadCtx parent_ctx;
    parent_ctx.blocks = &blocks;
    parent_ctx.manifest = &*parent_manifest;
    parent_ctx.history = &manifests;
    parent_ctx.max_depth = cfg.max_chain_depth;
    parent_ctx.topology = parent_topo ? &*parent_topo : nullptr;

    app::ReconstructedTensorSource base_src(parent_ctx, std::move(*parent_st_header));

    std::unordered_map<std::string, store::ParentTensorInfo> parent_info;
    for (const auto& [tensor, g] : parent_manifest->groups) {
        parent_info[tensor] = store::ParentTensorInfo{parent, g.chain_depth};
    }

    align::Matcher matcher(base_src, target, topo);
    auto report = matcher.run();
    if (!report) return std::unexpected(report.error());

    return store::plan_commit_groups(base_src, target, topo, *report, parent_info, blocks, cfg);
}

/// Store every tensor as its own Full singleton group — the only thing
/// possible with no base to diff against (root commit), and the fallback
/// when the real alignment path below fails for any reason. Reads by BYTE
/// range (read_raw), same as the original always-Full implementation, not
/// by output unit — a tensor's unit count is shape[0], which is 0 for a
/// genuinely scalar (0-dim) tensor, and read_units(name, 0, 0, ...) would
/// silently copy nothing for one.
core::Status commit_full_only(store::BlockStore& blocks, stio::StSource& source,
                              format::Manifest& manifest) {
    std::vector<std::byte> tbuf;
    for (const auto& entry : source.buffer_layout()) {
        tbuf.resize(static_cast<std::size_t>(entry.nbytes));
        auto n = source.read_raw(entry.off, tbuf);
        if (!n) return std::unexpected(n.error());
        if (*n != tbuf.size())
            return SFS_ERR(Io, "short read on tensor", entry.tensor);

        auto block_oid = blocks.put(core::ObjectKind::Raw, tbuf);
        if (!block_oid) return std::unexpected(block_oid.error());

        format::GroupEntry g;
        g.mode = format::GroupMode::Full;
        g.block = *block_oid;
        g.chain_depth = 0;
        manifest.groups[entry.tensor] = g;
    }
    return {};
}

int run_commit(const std::string& file, const std::string& message, const std::string& author_opt,
              const std::string& topology_path) {
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

    format::Manifest manifest;
    manifest.file.name = checkpoint_path.filename().string();
    manifest.file.header_block = *header_oid;
    manifest.file.total_bytes = (*source)->total_bytes();

    // Pass 1, always: the sha256 witness and the buffer layout need every
    // tensor's raw bytes regardless of how each group ends up stored. No
    // blocks.put() here — Full-only storage (commit_full_only, below) reads
    // tensors a second time rather than holding all of them in memory at
    // once (ADR 0008: a checkpoint is never fully resident).
    std::vector<std::byte> tbuf;
    for (const auto& entry : (*source)->buffer_layout()) {
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

        format::BufferEntry be;
        be.tensor = entry.tensor;
        be.off = entry.off;
        be.nbytes = entry.nbytes;
        be.group = entry.tensor;
        manifest.buffer.push_back(std::move(be));
    }
    manifest.file.sha256 = sha.finish_hex();

    auto topo_bytes_for_storage = load_topology_bytes(topology_path, checkpoint_path);
    if (!topo_bytes_for_storage) {
        std::cerr << "error: " << topo_bytes_for_storage.error().to_string() << "\n";
        return exit_code_for(topo_bytes_for_storage.error());
    }
    // "{}" (no real config given) parses as a hard error (missing 'layers'),
    // where a genuinely empty span takes parse_topology's documented
    // inference path instead — see load_topology_bytes's own comment.
    static constexpr char kPlaceholder[] = "{}";
    const bool has_real_topology =
        !(topo_bytes_for_storage->size() == 2 &&
         std::memcmp(topo_bytes_for_storage->data(), kPlaceholder, 2) == 0);
    std::span<const std::byte> parse_bytes =
        has_real_topology ? std::span<const std::byte>(*topo_bytes_for_storage)
                          : std::span<const std::byte>{};

    auto topo = align::parse_topology(**source, parse_bytes, align::ParseOptions{});
    if (!topo) {
        // Only the has_real_topology branch can fail (an empty span cannot,
        // per parse_topology's own doc) -- i.e. the user handed us a real
        // but malformed config.json. That is worth failing the commit over:
        // silently ignoring a config the user explicitly asked for would be
        // a worse surprise than an error naming the problem now.
        std::cerr << "error: " << topo.error().to_string() << "\n";
        return exit_code_for(topo.error());
    }

    bool planned = false;
    if (parent) {
        auto entries = plan_with_alignment(**blocks, commits, manifests, *parent, **source, *topo, *cfg);
        if (entries) {
            // format::Manifest::groups is a std::map (ordered); plan_commit_groups
            // returns std::unordered_map -- different container types, so this
            // has to move element-by-element rather than assign wholesale.
            for (auto& [name, g] : *entries) manifest.groups[name] = std::move(g);
            planned = true;
        } else {
            std::cerr << "warning: alignment failed (" << entries.error().to_string()
                      << "); committing every tensor Full\n";
        }
    }

    if (!planned) {
        if (auto st = commit_full_only(**blocks, **source, manifest); !st) {
            std::cerr << "error: " << st.error().to_string() << "\n";
            return exit_code_for(st.error());
        }
    }

    if (auto st = manifest.validate(); !st) {
        std::cerr << "error: built an invalid manifest: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }

    auto manifest_oid = manifests.write(manifest);
    if (!manifest_oid) {
        std::cerr << "error: " << manifest_oid.error().to_string() << "\n";
        return exit_code_for(manifest_oid.error());
    }

    auto topology_oid = (*blocks)->put(core::ObjectKind::Topology, *topo_bytes_for_storage);
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

    const auto n_delta = std::count_if(manifest.groups.begin(), manifest.groups.end(),
                                       [](const auto& kv) {
                                           return kv.second.mode == format::GroupMode::Delta;
                                       });
    std::cout << "[" << ref_name << " " << commit_oid->abbrev() << "] " << commit.message << "\n";
    std::cout << " " << manifest.buffer.size() << " tensor(s), " << manifest.file.total_bytes
              << " bytes";
    if (planned) std::cout << ", " << n_delta << " delta group(s)";
    std::cout << "\n";
    return ExitCode::Ok;
}

}  // namespace

void register_commit(CLI::App& app, int& exit_code) {
    static std::string file;
    static std::string message;
    static std::string author;
    static std::string topology;
    auto* c = app.add_subcommand("commit", "Store a .safetensors checkpoint as a new commit");
    c->add_option("file", file, "Path to the .safetensors checkpoint")->required();
    c->add_option("-m,--message", message, "Commit message");
    c->add_option("--author", author, "Author (default: $USER)");
    c->add_option("--topology", topology,
                  "Path to config.json describing the tensor topology "
                  "(default: <file's directory>/config.json if present)");
    c->callback([&exit_code] { exit_code = run_commit(file, message, author, topology); });
}

}  // namespace sfs::app::cmd
