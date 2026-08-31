/// \file daemon.cpp
/// Session lifecycle around the low-level FUSE adapter in fuse_ll.cpp.
///
/// This file owns exactly three things: building `fuse_args`/the session,
/// running libfuse's own loop ("libfuse owns the threads" -- prefetch.hpp,
/// thread_pool.hpp), and the repo-side mount registration that makes `gc`
/// refuse while attached (docs/spec/11-repo-layout.md §6, store/gc.hpp).
/// Everything inode/read-path related lives in fuse_ll.cpp and fs.cpp.

#include <synapsefs/mount/daemon.hpp>

#include "fuse_ll_internal.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <synapsefs/util/atomic_io.hpp>
#include <synapsefs/util/file.hpp>
#include <synapsefs/util/log.hpp>
#include <synapsefs/util/scope_guard.hpp>

extern char** environ;

namespace sfs::mount {

namespace {
constexpr std::string_view kLogModule = "mount.daemon";

// Repo-side marker recording that a daemon is attached. RepoPaths (core/
// repo_config.hpp) doesn't expose a dedicated accessor for this -- it only
// names object/ref/journal/lock paths -- so this lives under the standard
// `.synapsefs` directory returned by RepoPaths::dot(), namespaced clearly
// enough not to collide with anything the store module owns. store::gc's
// `mount_attached()` (store/gc.hpp) is expected to check for exactly this
// file; the two are two sides of one small contract.
std::filesystem::path mount_marker_path(const core::RepoPaths& paths) {
    return paths.dot() / "mount-daemon.pid";
}

std::string build_marker_contents(const std::filesystem::path& mountpoint) {
    return std::to_string(static_cast<long long>(::getpid())) + "\n" +
           mountpoint.string() + "\n";
}
}  // namespace

struct Daemon::Impl {
    detail::MountCtx        mctx;
    std::unique_ptr<SynapseFs> fs;   // mctx.fs points into this
    DaemonOptions            opts;

    fuse_args        args = FUSE_ARGS_INIT(0, nullptr);
    fuse_session*    session = nullptr;
    bool             mounted = false;

    std::optional<core::RepoPaths> registered_repo;  // set by register_with_repo

    ~Impl() {
        if (session != nullptr) {
            if (mounted) fuse_session_unmount(session);
            fuse_session_destroy(session);
        }
        fuse_opt_free_args(&args);

        if (registered_repo) {
            std::error_code ec;
            std::filesystem::remove(mount_marker_path(*registered_repo), ec);
            // Best-effort: an unremoved marker just makes the next `gc`
            // report "attached" until an operator clears it, which is a
            // safe failure mode (refuse-to-collect), not a corrupting one.
        }
    }
};

Daemon::Daemon() : impl_(std::make_unique<Impl>()) {}
Daemon::~Daemon() = default;

core::Result<std::unique_ptr<Daemon>> Daemon::start(std::unique_ptr<SynapseFs> fs,
                                                     DaemonOptions options) {
    if (fs == nullptr) {
        return std::unexpected(
            core::make_error(core::ErrKind::Internal, "Daemon::start: null filesystem"));
    }

    auto daemon = std::unique_ptr<Daemon>(new Daemon());
    Daemon::Impl& im = *daemon->impl_;

    im.fs   = std::move(fs);
    im.opts = std::move(options);

    im.mctx.fs        = im.fs.get();
    im.mctx.mount_time = std::time(nullptr);
    im.mctx.max_read   = im.opts.fs.max_read;

    // fuse_args wants an argv-shaped list. We don't take user-supplied FUSE
    // mount options here (allow_other is the one knob DaemonOptions/FsOptions
    // exposes; anything more exotic is a repo-config concern, not this
    // call's), so this is a small, fixed argument vector.
    std::vector<std::string> argv_storage;
    argv_storage.push_back("sfs-mount");  // argv[0]; libfuse only uses it for -h/-V text
    if (im.opts.debug) argv_storage.push_back("-d");
    if (im.opts.fs.allow_other) argv_storage.push_back("-oallow_other");

    std::vector<char*> argv;
    argv.reserve(argv_storage.size());
    for (auto& s : argv_storage) argv.push_back(s.data());
    im.args = FUSE_ARGS_INIT(static_cast<int>(argv.size()), argv.data());

    im.session = fuse_session_new(&im.args, &detail::ll_ops(), sizeof(detail::ll_ops()), &im.mctx);
    if (im.session == nullptr) {
        return std::unexpected(
            core::make_error(core::ErrKind::MountFailed, "fuse_session_new failed"));
    }

    if (fuse_set_signal_handlers(im.session) != 0) {
        SFS_LOG_W(kLogModule, "fuse_set_signal_handlers failed; Ctrl-C won't clean up the mount");
    }

    std::error_code mkec;
    std::filesystem::create_directories(im.opts.mountpoint, mkec);
    if (mkec) {
        return std::unexpected(core::make_error(core::ErrKind::MountFailed,
                                                 "cannot create mountpoint: " + mkec.message(),
                                                 im.opts.mountpoint.string()));
    }

    if (fuse_session_mount(im.session, im.opts.mountpoint.c_str()) != 0) {
        return std::unexpected(core::make_error(core::ErrKind::MountFailed,
                                                 "fuse_session_mount failed",
                                                 im.opts.mountpoint.string()));
    }
    im.mounted = true;

    SFS_LOG_I(kLogModule, "mounted {} at {}", im.fs->file_name(), im.opts.mountpoint.string());
    return daemon;
}

core::Status Daemon::run() {
    Impl& im = *impl_;
    if (im.session == nullptr || !im.mounted) {
        return std::unexpected(
            core::make_error(core::ErrKind::MountFailed, "Daemon::run: not mounted"));
    }

    // Multi-threaded: "libfuse owns the threads" (prefetch.hpp,
    // util/thread_pool.hpp) is a design decision, not just a comment --
    // concurrent readers are what test_blockcache_race.cpp and
    // tests/concurrent_readers.cpp actually exercise, and single-flight fill
    // is only an interesting property under real concurrency.
#ifdef fuse_loop_cfg_create
    struct fuse_loop_config* cfg = fuse_loop_cfg_create();
    SFS_DEFER { if (cfg != nullptr){ fuse_loop_cfg_destroy(cfg);} };
    if (cfg != nullptr) {
        fuse_loop_cfg_set_clone_fd(cfg, 0);
        fuse_loop_cfg_set_idle_threads(cfg, 10);
    }
    const int rc = fuse_session_loop_mt(im.session, cfg);
#else
    const int rc = fuse_session_loop(im.session);
#endif

    if (fuse_session_exited(im.session) && rc == 0) {
        return core::Status{};
    }
    if (rc != 0) {
        return std::unexpected(core::make_error(
            core::ErrKind::MountFailed,
            "fuse_session_loop_mt failed (rc=" + std::to_string(rc) + ")",
            im.opts.mountpoint.string()));
    }
    return core::Status{};
}

void Daemon::stop() noexcept {
    Impl& im = *impl_;
    if (im.session != nullptr) {
        fuse_session_exit(im.session);
        // Nudge the loop past its poll/read if it's blocked with nothing
        // in flight; fuse_session_exit alone only sets a flag the next
        // request notices.
        fuse_session_unmount(im.session);
        im.mounted = false;
    }
}

core::Status Daemon::register_with_repo(const core::RepoPaths& paths) {
    Impl& im = *impl_;

    std::error_code ec;
    std::filesystem::create_directories(paths.dot(), ec);
    if (ec) {
        return std::unexpected(core::from_errno(ec, paths.dot().string()));
    }

    const std::string contents = build_marker_contents(im.opts.mountpoint);
    const std::filesystem::path marker = mount_marker_path(paths);

    // Atomic, crash-safe write, same idiom the rest of the repo uses for
    // anything that must never be observed half-written (util/atomic_io.hpp:
    // temp file, fsync, rename, fsync(parent) -- ADR 0007). util speaks
    // errno, not sfs::core::Error (util deliberately doesn't depend on
    // core), so the boundary conversion is ours to do here.
    util::AtomicWriteOptions write_opts;
    write_opts.temp_dir = paths.tmp();
    auto wrote = util::atomic_write(
        marker, std::as_bytes(std::span(contents.data(), contents.size())), write_opts);
    if (!wrote) {
        return std::unexpected(core::from_errno(wrote.error(), marker.string()));
    }

    im.registered_repo = paths;
    SFS_LOG_I(kLogModule, "registered mount with repo at {}", paths.root.string());
    return {};
}

core::Status unmount(const std::filesystem::path& mountpoint) {
    // "Waits for in-flight reads, falls back to fusermount3 -u" (daemon.hpp).
    // A graceful unmount from a SEPARATE process (this is `sfs unmount`, not
    // the daemon's own stop()) has no fuse_session to signal -- fusermount3
    // is the only portable way in, and it itself waits for the kernel to
    // drain in-flight requests before it returns.
    const std::string mp = mountpoint.string();

    pid_t pid = -1;
    std::vector<const char*> argv = {"fusermount3", "-u", mp.c_str(), nullptr};
    // posix_spawnp searches PATH, matching how the demo scripts invoke it.
    int rc = ::posix_spawnp(&pid, "fusermount3", nullptr, nullptr,
                             const_cast<char* const*>(argv.data()), environ);
    if (rc != 0) {
        return std::unexpected(core::from_errno(std::error_code(rc, std::generic_category()),
                                                 "spawning fusermount3"));
    }

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        return std::unexpected(
            core::from_errno(std::error_code(errno, std::generic_category()), mp));
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return std::unexpected(core::make_error(
            core::ErrKind::MountFailed, "fusermount3 -u failed", mp));
    }
    return {};
}

}  // namespace sfs::mount
