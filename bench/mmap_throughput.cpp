/// bench/mmap_throughput.cpp — docs/benchmarks.md §4.
///
/// Commits a checkpoint into a scratch repo (same construction as
/// verify_time.cpp), mounts it for real via mount::Daemon, and measures four
/// read paths against the FUSE-mounted file: sequential read(), mmap'd
/// sequential page-touch, random 4 KiB pread(), and an ext4 baseline reading
/// the same bytes straight off the original file (no FUSE).
///
/// NOT measured here, honestly:
///   - cold cache (`--full` style cache-dropping needs `sudo` + drop_caches,
///     which this tool won't run non-interactively; every number below is
///     warm-cache)
///   - the `load_file()` row in docs/benchmarks.md §4 -- that's Python +
///     torch loading through the mount, a different tool's job, not this one
///   - concurrent/multi-depth random reads ("depth 5" in the docs table) --
///     this only drives depth-1 (synchronous) random reads
///
/// Same rationale as verify_time.cpp for living outside apps/sfs's CLI:
/// mount/store/codec/format/core/stio build without Torch, so this doesn't
/// need align at all.

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include <synapsefs/core/oid.hpp>
#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/format/commit.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/mount/daemon.hpp>
#include <synapsefs/mount/fs.hpp>
#include <synapsefs/stio/st_source.hpp>
#include <synapsefs/stio/st_writer.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/manifest_store.hpp>
#include <synapsefs/store/refs.hpp>

using namespace sfs;
namespace fs = std::filesystem;
using json = nlohmann::json;
using clk = std::chrono::steady_clock;

namespace {

double secs_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

struct Stats {
    double throughput_mb_s = 0;
    double p50_us = 0;
    double p99_us = 0;
};

Stats summarize(std::vector<double>& latencies_us, std::size_t bytes, double wall_s) {
    Stats s;
    s.throughput_mb_s = (bytes / 1e6) / wall_s;
    if (!latencies_us.empty()) {
        std::sort(latencies_us.begin(), latencies_us.end());
        s.p50_us = latencies_us[latencies_us.size() * 50 / 100];
        s.p99_us = latencies_us[std::min(latencies_us.size() - 1, latencies_us.size() * 99 / 100)];
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::optional<std::string> checkpoint_arg;
    int random_reads = 500;
    bool as_json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--checkpoint" && i + 1 < argc) checkpoint_arg = argv[++i];
        else if (arg == "--reads" && i + 1 < argc) random_reads = std::atoi(argv[++i]);
        else if (arg == "--json") as_json = true;
    }
    if (!checkpoint_arg) {
        fprintf(stderr, "usage: mmap_throughput --checkpoint <path.safetensors> [--reads N] [--json]\n");
        return 2;
    }
    const fs::path checkpoint(*checkpoint_arg);

    // ---- set up a real repo, commit the checkpoint, exactly like
    // verify_time.cpp / apps/sfs/cmd/commit.cpp ----
    const fs::path root = fs::temp_directory_path() / "sfs_mmap_bench";
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
        fprintf(stderr, "%s\n", st.error().to_string().c_str());
        return 1;
    }
    store::RefStore refs(paths);
    if (auto st = refs.set_head_symbolic("refs/heads/main"); !st) {
        fprintf(stderr, "%s\n", st.error().to_string().c_str());
        return 1;
    }
    auto blocks_r = store::BlockStore::open(paths, cfg);
    if (!blocks_r) { fprintf(stderr, "%s\n", blocks_r.error().to_string().c_str()); return 1; }
    auto& blocks = **blocks_r;
    store::CommitStore commits(blocks, refs);
    store::ManifestStore manifests(blocks, commits);

    auto source = stio::StSource::open(checkpoint);
    if (!source) { fprintf(stderr, "%s\n", source.error().to_string().c_str()); return 1; }

    format::Manifest manifest;
    manifest.file.name = checkpoint.filename().string();
    stio::Sha256Stream sha;
    sha.update((*source)->header_bytes());
    auto header_oid = blocks.put(core::ObjectKind::Header, (*source)->header_bytes());
    if (!header_oid) { fprintf(stderr, "%s\n", header_oid.error().to_string().c_str()); return 1; }
    manifest.file.header_block = *header_oid;
    manifest.file.total_bytes = (*source)->total_bytes();

    std::vector<std::byte> tbuf;
    for (const auto& entry : (*source)->buffer_layout()) {
        tbuf.resize(static_cast<std::size_t>(entry.nbytes));
        auto n = (*source)->read_raw(entry.off, tbuf);
        if (!n || *n != tbuf.size()) {
            fprintf(stderr, "short read: %s\n", entry.tensor.c_str());
            return 1;
        }
        sha.update(tbuf);
        auto block_oid = blocks.put(core::ObjectKind::Raw, tbuf);
        if (!block_oid) { fprintf(stderr, "%s\n", block_oid.error().to_string().c_str()); return 1; }
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
    if (auto st = manifest.validate(); !st) { fprintf(stderr, "%s\n", st.error().to_string().c_str()); return 1; }
    auto manifest_oid = manifests.write(manifest);
    if (!manifest_oid) { fprintf(stderr, "%s\n", manifest_oid.error().to_string().c_str()); return 1; }

    // No real topology to reference honestly here either -- same "{}"
    // placeholder apps/sfs/cmd/commit.cpp's store_topology() writes today.
    static constexpr char kEmptyTopo[] = "{}";
    auto topology_oid = blocks.put(core::ObjectKind::Topology,
                                   std::span(reinterpret_cast<const std::byte*>(kEmptyTopo), 2));
    if (!topology_oid) { fprintf(stderr, "%s\n", topology_oid.error().to_string().c_str()); return 1; }

    format::Commit commit;
    commit.manifest = *manifest_oid;
    commit.topology = *topology_oid;
    commit.timestamp = format::now_timestamp();
    commit.author = "bench";
    commit.message = "mmap_throughput bench commit";
    auto commit_oid = commits.commit_and_advance(commit, "refs/heads/main", std::nullopt);
    if (!commit_oid) { fprintf(stderr, "%s\n", commit_oid.error().to_string().c_str()); return 1; }
    if (!as_json) {
        printf("committed %s (%llu bytes, %zu tensors)\n", commit_oid->abbrev().c_str(),
              (unsigned long long)manifest.file.total_bytes, manifest.buffer.size());
    }

    // ---- mount it for real ----
    codec::ReadCtx ctx;
    ctx.blocks = &blocks;
    ctx.manifest = &manifest;
    ctx.history = &manifests;
    ctx.max_depth = cfg.max_chain_depth;

    mount::FsOptions fs_opts;
    fs_opts.cache_bytes = cfg.cache_bytes;
    auto fs_obj = mount::SynapseFs::create(ctx, manifest, fs_opts);
    if (!fs_obj) { fprintf(stderr, "%s\n", fs_obj.error().to_string().c_str()); return 1; }

    const fs::path mountpoint = fs::temp_directory_path() / "sfs_mmap_bench_mnt";
    fs::remove_all(mountpoint, ec);
    fs::create_directories(mountpoint, ec);

    mount::DaemonOptions dopts;
    dopts.mountpoint = mountpoint;
    dopts.foreground = true;
    dopts.fs = fs_opts;
    auto daemon = mount::Daemon::start(std::move(*fs_obj), dopts);
    if (!daemon) { fprintf(stderr, "mount failed: %s\n", daemon.error().to_string().c_str()); return 1; }

    std::thread runner([&] { (void)(*daemon)->run(); });

    const fs::path mounted_file = mountpoint / manifest.file.name;
    bool ready = false;
    for (int i = 0; i < 100 && !ready; ++i) {
        std::error_code sec;
        if (fs::exists(mounted_file, sec) && fs::file_size(mounted_file, sec) == manifest.file.total_bytes)
            ready = true;
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!ready) {
        fprintf(stderr, "mounted file never appeared at %s\n", mounted_file.string().c_str());
        mount::unmount(mountpoint);
        runner.join();
        return 1;
    }

    const std::size_t total_bytes = static_cast<std::size_t>(manifest.file.total_bytes);

    double seq_read_mb_s = 0, mmap_mb_s = 0, ext4_mb_s = 0;
    bool mmap_failed = false;
    Stats random_stats;

    // 1. Sequential read, whole file.
    {
        int fd = ::open(mounted_file.c_str(), O_RDONLY);
        std::vector<std::byte> buf(4u << 20);
        auto t0 = clk::now();
        std::size_t done = 0;
        while (done < total_bytes) {
            ssize_t n = ::read(fd, buf.data(), std::min(buf.size(), total_bytes - done));
            if (n <= 0) break;
            done += static_cast<std::size_t>(n);
        }
        const double s = secs_since(t0);
        ::close(fd);
        seq_read_mb_s = (done / 1e6) / s;
    }

    // 2. mmap sequential (page touch).
    {
        int fd = ::open(mounted_file.c_str(), O_RDONLY);
        void* p = ::mmap(nullptr, total_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            mmap_failed = true;
        } else {
            volatile std::uint8_t sink = 0;
            auto t0 = clk::now();
            const auto* bytes = static_cast<const std::uint8_t*>(p);
            constexpr std::size_t kPage = 4096;
            for (std::size_t off = 0; off < total_bytes; off += kPage) sink ^= bytes[off];
            const double s = secs_since(t0);
            mmap_mb_s = (total_bytes / 1e6) / s;
            ::munmap(p, total_bytes);
        }
        ::close(fd);
    }

    // 3. Random 4 KiB, depth 0 (single-threaded, synchronous pread).
    {
        int fd = ::open(mounted_file.c_str(), O_RDONLY);
        constexpr std::size_t kChunk = 4096;
        const std::size_t n_chunks = total_bytes / kChunk;
        std::mt19937_64 rng(3);
        std::uniform_int_distribution<std::size_t> dist(0, n_chunks > 0 ? n_chunks - 1 : 0);
        std::vector<std::byte> buf(kChunk);
        std::vector<double> lat_us;
        auto t0 = clk::now();
        for (int i = 0; i < random_reads && n_chunks > 0; ++i) {
            const std::size_t off = dist(rng) * kChunk;
            auto r0 = clk::now();
            ::pread(fd, buf.data(), kChunk, static_cast<off_t>(off));
            lat_us.push_back(std::chrono::duration<double, std::micro>(clk::now() - r0).count());
        }
        const double s = secs_since(t0);
        ::close(fd);
        random_stats = summarize(lat_us, static_cast<std::size_t>(random_reads) * kChunk, s);
    }

    // 4. Baseline: same bytes, read straight off the real filesystem (no
    // FUSE) -- meaningful as a same-medium comparison as long as `checkpoint`
    // and the repo/mountpoint scratch dirs above sit on the same filesystem
    // (both default to fs::temp_directory_path(); pass a --checkpoint on a
    // different mount and this stops being apples-to-apples).
    {
        int fd = ::open(checkpoint.c_str(), O_RDONLY);
        std::vector<std::byte> buf(4u << 20);
        auto t0 = clk::now();
        std::size_t done = 0;
        ssize_t n;
        while ((n = ::read(fd, buf.data(), buf.size())) > 0) done += static_cast<std::size_t>(n);
        const double s = secs_since(t0);
        ::close(fd);
        ext4_mb_s = (done / 1e6) / s;
    }

    mount::unmount(mountpoint);
    runner.join();

    if (as_json) {
        json j = {
            {"total_bytes", total_bytes},
            {"sequential_read_mb_s", seq_read_mb_s},
            {"mmap_sequential_mb_s", mmap_failed ? json(nullptr) : json(mmap_mb_s)},
            {"random_4kib", {{"reads", random_reads},
                             {"mb_s", random_stats.throughput_mb_s},
                             {"p50_us", random_stats.p50_us},
                             {"p99_us", random_stats.p99_us}}},
            {"baseline_same_file_no_fuse_mb_s", ext4_mb_s},
            {"cache_state", "warm"},
        };
        printf("%s\n", j.dump(2).c_str());
    } else {
        printf("\n=== mmap_throughput (WARM cache -- no sudo drop_caches run) ===\n");
        printf("Sequential read, whole file:   %.1f MB/s\n", seq_read_mb_s);
        if (mmap_failed) {
            printf("mmap sequential (page touch):  FAILED\n");
        } else {
            printf("mmap sequential (page touch):  %.1f MB/s\n", mmap_mb_s);
        }
        printf("Random 4 KiB, depth 0:         %.1f MB/s, p50=%.1f us, p99=%.1f us\n",
              random_stats.throughput_mb_s, random_stats.p50_us, random_stats.p99_us);
        printf("Baseline, same file on ext4:    %.1f MB/s\n", ext4_mb_s);
    }
    return 0;
}
