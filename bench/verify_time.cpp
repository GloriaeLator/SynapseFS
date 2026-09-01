/// bench/verify_time.cpp — docs/benchmarks.md §3.
///
/// Builds a small repo directly via store:: APIs (BlockStore/CommitStore/
/// ManifestStore/RefStore — the same calls apps/sfs/cmd/init.cpp and
/// commit.cpp make) rather than shelling out to a real `sfs init`/`sfs
/// commit`, and times store::verify() against it. Also a --hash-only mode
/// comparing BLAKE3 (core::digest) against SHA-256 (stio::Sha256Stream) on a
/// large random buffer.
///
/// Deliberately NOT wired through apps/sfs's CLI: this needs only
/// core/format/stio/store, none of which need align/Torch to build, whereas
/// the `sfs` binary links align in full for real topology loading
/// (apps/sfs/header_only_source.hpp's load_commit_topology). Until that's
/// resolved, `bench/CMakeLists.txt`'s blanket `synapsefs::align` link on
/// every bench target would block this one for a dependency it never
/// actually needs — same situation residual_codec.cpp's bench already
/// documents.
///
/// The docs/benchmarks.md command line this replaces
/// (`verify_time --repo <repo> --json`) assumed a pre-existing on-disk repo;
/// this tool builds one itself from a checkpoint instead, since there is no
/// real `sfs commit` to have produced one with yet.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <synapsefs/core/oid.hpp>
#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/format/commit.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/stio/st_source.hpp>
#include <synapsefs/stio/st_writer.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/manifest_store.hpp>
#include <synapsefs/store/refs.hpp>
#include <synapsefs/store/verify.hpp>

using namespace sfs;
namespace fs = std::filesystem;
using json = nlohmann::json;
using clk = std::chrono::steady_clock;

namespace {

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// "1G" / "512M" / "4096" -- the shape docs/benchmarks.md's own example
// command (`--bytes 1G`) uses.
std::size_t parse_byte_size(const std::string& s) {
    if (s.empty()) return 0;
    char suffix = s.back();
    std::uint64_t mult = 1;
    std::string digits = s;
    if (suffix == 'G' || suffix == 'g') { mult = 1ull << 30; digits.pop_back(); }
    else if (suffix == 'M' || suffix == 'm') { mult = 1ull << 20; digits.pop_back(); }
    else if (suffix == 'K' || suffix == 'k') { mult = 1ull << 10; digits.pop_back(); }
    return static_cast<std::size_t>(std::stoull(digits) * mult);
}

core::Oid commit_checkpoint(store::BlockStore& blocks, store::CommitStore& commits,
                            store::ManifestStore& manifests, const fs::path& checkpoint,
                            std::optional<core::Oid> parent) {
    auto source = stio::StSource::open(checkpoint);
    if (!source) throw std::runtime_error(source.error().to_string());

    stio::Sha256Stream sha;
    sha.update((*source)->header_bytes());

    auto header_oid = blocks.put(core::ObjectKind::Header, (*source)->header_bytes());
    if (!header_oid) throw std::runtime_error(header_oid.error().to_string());

    format::Manifest manifest;
    manifest.file.name = checkpoint.filename().string();
    manifest.file.header_block = *header_oid;
    manifest.file.total_bytes = (*source)->total_bytes();

    std::vector<std::byte> tbuf;
    for (const auto& entry : (*source)->buffer_layout()) {
        tbuf.resize(static_cast<std::size_t>(entry.nbytes));
        auto n = (*source)->read_raw(entry.off, tbuf);
        if (!n || *n != tbuf.size()) throw std::runtime_error("short read: " + entry.tensor);
        sha.update(tbuf);

        auto block_oid = blocks.put(core::ObjectKind::Raw, tbuf);
        if (!block_oid) throw std::runtime_error(block_oid.error().to_string());

        format::GroupEntry g;
        g.mode = format::GroupMode::Full;
        g.block = *block_oid;
        manifest.groups[entry.tensor] = g;

        format::BufferEntry be;
        be.tensor = entry.tensor;
        be.off = entry.off;
        be.nbytes = entry.nbytes;
        be.group = entry.tensor;
        manifest.buffer.push_back(std::move(be));
    }
    manifest.file.sha256 = sha.finish_hex();

    if (auto st = manifest.validate(); !st) throw std::runtime_error(st.error().to_string());
    auto manifest_oid = manifests.write(manifest);
    if (!manifest_oid) throw std::runtime_error(manifest_oid.error().to_string());

    // No topology object exists to reference honestly yet (align::Matcher
    // can't build here either), so this stores the same "{}" placeholder
    // apps/sfs/cmd/commit.cpp's store_topology() does today.
    static constexpr char kEmptyTopo[] = "{}";
    auto topology_oid = blocks.put(core::ObjectKind::Topology,
                                   std::span(reinterpret_cast<const std::byte*>(kEmptyTopo), 2));
    if (!topology_oid) throw std::runtime_error(topology_oid.error().to_string());

    format::Commit commit;
    if (parent) commit.parents = {*parent};
    commit.manifest = *manifest_oid;
    commit.topology = *topology_oid;
    commit.timestamp = format::now_timestamp();
    commit.author = "bench";
    commit.message = "verify_time bench commit";

    auto commit_oid = commits.commit_and_advance(commit, "refs/heads/main", parent);
    if (!commit_oid) throw std::runtime_error(commit_oid.error().to_string());
    return *commit_oid;
}

void run_hash_only(std::size_t n_bytes, bool as_json) {
    std::vector<std::byte> buf(n_bytes);
    std::mt19937_64 rng(1);
    for (std::size_t i = 0; i < buf.size(); i += 8) {
        std::uint64_t v = rng();
        std::memcpy(buf.data() + i, &v, std::min<std::size_t>(8, buf.size() - i));
    }

    const auto t0 = clk::now();
    (void)core::digest(buf);
    const double blake3_s = std::chrono::duration<double>(clk::now() - t0).count();
    const double blake3_gbs = (n_bytes / 1e9) / blake3_s;

    stio::Sha256Stream sha;
    const auto t1 = clk::now();
    sha.update(buf);
    (void)sha.finish_hex();
    const double sha_s = std::chrono::duration<double>(clk::now() - t1).count();
    const double sha_gbs = (n_bytes / 1e9) / sha_s;

    if (as_json) {
        json j = {{"bytes", n_bytes},
                 {"blake3_1thread_gbs", blake3_gbs},
                 {"sha256_gbs", sha_gbs}};
        printf("%s\n", j.dump(2).c_str());
    } else {
        printf("BLAKE3, 1 thread: %.3f GB/s\n", blake3_gbs);
        printf("SHA-256:          %.3f GB/s\n", sha_gbs);
        printf("(BLAKE3, all cores: not measured -- core::Hasher/core::digest only expose "
              "single-threaded hashing)\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::optional<std::string> checkpoint;
    int num_commits = 5;
    bool hash_only = false;
    std::size_t hash_bytes = 1ull << 30;
    bool as_json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--checkpoint" && i + 1 < argc) checkpoint = argv[++i];
        else if (arg == "--commits" && i + 1 < argc) num_commits = std::atoi(argv[++i]);
        else if (arg == "--hash-only") hash_only = true;
        else if (arg == "--bytes" && i + 1 < argc) hash_bytes = parse_byte_size(argv[++i]);
        else if (arg == "--json") as_json = true;
    }

    if (hash_only) {
        run_hash_only(hash_bytes, as_json);
        return 0;
    }
    if (!checkpoint) {
        fprintf(stderr,
               "usage: verify_time --checkpoint <path.safetensors> [--commits N] [--json]\n"
               "       verify_time --hash-only [--bytes 1G] [--json]\n");
        return 2;
    }

    const fs::path root = fs::temp_directory_path() / "sfs_verify_time_bench";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    core::RepoPaths paths{root};
    fs::create_directories(paths.objects(), ec);
    fs::create_directories(paths.tmp(), ec);
    fs::create_directories(paths.refs_heads(), ec);
    fs::create_directories(paths.journal(), ec);

    core::RepoConfig cfg;
    if (auto st = cfg.save(root); !st) {
        fprintf(stderr, "cfg.save: %s\n", st.error().to_string().c_str());
        return 1;
    }

    store::RefStore refs(paths);
    if (auto st = refs.set_head_symbolic("refs/heads/main"); !st) {
        fprintf(stderr, "%s\n", st.error().to_string().c_str());
        return 1;
    }

    auto blocks_r = store::BlockStore::open(paths, cfg);
    if (!blocks_r) {
        fprintf(stderr, "%s\n", blocks_r.error().to_string().c_str());
        return 1;
    }
    auto& blocks = **blocks_r;
    store::CommitStore commits(blocks, refs);
    store::ManifestStore manifests(blocks, commits);

    std::optional<core::Oid> parent;
    core::Oid tip;
    try {
        for (int i = 0; i < num_commits; ++i) {
            tip = commit_checkpoint(blocks, commits, manifests, *checkpoint, parent);
            parent = tip;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "commit failed: %s\n", e.what());
        return 1;
    }

    const std::array<core::Oid, 1> tips{tip};

    // Quick verify: structural + ancestor-invariant checks, chunk-level
    // integrity NOT re-hashed.
    auto t0 = clk::now();
    auto report = store::verify(blocks, commits, manifests, refs, tips, store::VerifyOptions{});
    const double quick_ms = ms_since(t0);
    if (!report) {
        fprintf(stderr, "verify failed: %s\n", report.error().to_string().c_str());
        return 1;
    }

    // Full verify: every chunk of every reachable object re-hashed.
    store::VerifyOptions full_opts;
    full_opts.full = true;
    auto t1 = clk::now();
    auto report_full = store::verify(blocks, commits, manifests, refs, tips, full_opts);
    const double full_ms = ms_since(t1);
    if (!report_full) {
        fprintf(stderr, "verify --full failed: %s\n", report_full.error().to_string().c_str());
        return 1;
    }

    if (as_json) {
        json j = {
            {"commits", num_commits},
            {"tip", tip.abbrev()},
            {"objects_checked", report->objects_checked},
            {"ok", report->ok()},
            {"findings", report->findings.size()},
            {"verify_ms", quick_ms},
            {"verify_full_ms", full_ms},
            {"bytes_hashed_full", report_full->bytes_hashed},
        };
        printf("%s\n", j.dump(2).c_str());
    } else {
        printf("committed %d commits, tip=%s\n", num_commits, tip.abbrev().c_str());
        printf("objects_checked=%llu findings=%zu ok=%d\n",
              (unsigned long long)report->objects_checked, report->findings.size(), report->ok());
        printf("verify:        %.3f ms\n", quick_ms);
        printf("verify --full: %.3f ms  (bytes_hashed=%llu)\n", full_ms,
              (unsigned long long)report_full->bytes_hashed);
    }
    return 0;
}
