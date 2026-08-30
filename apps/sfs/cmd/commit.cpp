/// `sfs commit <file>` — stores a .safetensors checkpoint as a new commit.
///
/// This build does not yet have an alignment engine (modules/align has no
/// .cpp), so every tensor is stored as its own singleton Full group — one
/// group per tensor, block = the tensor's raw bytes verbatim. No Delta group
/// is ever produced here, matching codec::reconstruct_file's documented
/// assumption. Wiring a real diff path in is modules/align + modules/codec's
/// job; this command's manifest-building logic does not need to change when
/// that lands — only how each group's `block`/`base` gets filled in.

#include <CLI/CLI.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>

#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/format/commit.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/stio/st_source.hpp>
#include <synapsefs/stio/st_writer.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/lockfile.hpp>
#include <synapsefs/store/manifest_store.hpp>
#include <synapsefs/store/refs.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {

namespace {

/// Store the topology object for this commit. Real topology parsing
/// (config.json -> core::Topology) is align::topology_parser's job and is
/// not wired into this build yet, so we store whatever bytes are available
/// verbatim and opaque — the block store does not interpret them. A commit
/// made under this policy is trivially re-committable once alignment lands;
/// nothing about the manifest/commit shape changes.
core::Result<core::Oid> store_topology(store::BlockStore& blocks,
                                       const std::string& topology_path,
                                       const std::filesystem::path& checkpoint_path) {
    std::filesystem::path candidate = topology_path.empty()
        ? checkpoint_path.parent_path() / "config.json"
        : std::filesystem::path(topology_path);

    std::error_code ec;
    if (!topology_path.empty() && !std::filesystem::exists(candidate, ec)) {
        return SFS_ERR(NoSuchFile, "topology file not found", candidate.string());
    }

    std::vector<std::byte> bytes;
    if (std::filesystem::exists(candidate, ec)) {
        std::ifstream f(candidate, std::ios::binary);
        f.seekg(0, std::ios::end);
        auto sz = f.tellg();
        f.seekg(0);
        bytes.resize(static_cast<std::size_t>(sz));
        f.read(reinterpret_cast<char*>(bytes.data()), sz);
    } else {
        // No config.json alongside the checkpoint and none given: placeholder
        // empty object rather than a null/dangling oid. Commit::topology is
        // not optional (docs/spec/10 §3) — a repo that has never seen a real
        // topology still needs every commit to satisfy that shape.
        static constexpr char kEmpty[] = "{}";
        bytes.assign(reinterpret_cast<const std::byte*>(kEmpty),
                    reinterpret_cast<const std::byte*>(kEmpty) + 2);
    }

    return blocks.put(core::ObjectKind::Topology, bytes);
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

        auto block_oid = (*blocks)->put(core::ObjectKind::Raw, tbuf);
        if (!block_oid) {
            std::cerr << "error: " << block_oid.error().to_string() << "\n";
            return exit_code_for(block_oid.error());
        }

        format::GroupEntry g;
        g.mode = format::GroupMode::Full;
        g.block = *block_oid;
        g.chain_depth = 0;
        manifest.groups[entry.tensor] = g;

        format::BufferEntry be;
        be.tensor = entry.tensor;
        be.off = entry.off;
        be.nbytes = entry.nbytes;
        be.group = entry.tensor;
        manifest.buffer.push_back(std::move(be));
    }
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

    auto topology_oid = store_topology(**blocks, topology_path, checkpoint_path);
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

    std::cout << "[" << ref_name << " " << commit_oid->abbrev() << "] " << commit.message << "\n";
    std::cout << " " << manifest.buffer.size() << " tensor(s), "
              << manifest.file.total_bytes << " bytes\n";
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
