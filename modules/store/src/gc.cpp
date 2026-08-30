#include <synapsefs/store/gc.hpp>

#include <fstream>
#include <string>
#include <unordered_set>

#include <cerrno>
#include <csignal>

#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/lockfile.hpp>
#include <synapsefs/store/manifest_store.hpp>

namespace sfs::store {

namespace {

/// The repo-side marker mount::Daemon::register_with_repo writes. Two sides
/// of one contract (see the comment in mount/src/daemon.cpp); the path is
/// duplicated rather than shared because store must not depend on mount.
std::filesystem::path mount_marker_path(const core::RepoPaths& paths) {
    return paths.dot() / "mount-daemon.pid";
}

/// Remove leftover temp files from crashed writes. These are unreferenced by
/// construction (atomic_write renames into place; anything still in tmp/
/// never got that far), so this is safe whenever the exclusive lock is held.
std::uint64_t sweep_temp_dir(const std::filesystem::path& dir, bool dry_run,
                             std::uint64_t& bytes_reclaimed) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return 0;

    std::uint64_t removed = 0;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        auto sz = it->file_size(ec);
        if (ec) {
            ec.clear();
            continue;
        }
        if (!dry_run) {
            std::error_code rm_ec;
            std::filesystem::remove(it->path(), rm_ec);
            if (rm_ec) continue;
        }
        bytes_reclaimed += sz;
        ++removed;
    }
    return removed;
}

}  // namespace

Result<bool> mount_attached(const core::RepoPaths& paths) {
    const auto marker = mount_marker_path(paths);

    std::error_code ec;
    if (!std::filesystem::exists(marker, ec)) return false;

    // A marker whose process is gone is stale: a daemon that was killed
    // (-9, crash, reboot) never got to remove it. Treating stale markers as
    // "attached" would make gc permanently refuse after one crashed mount,
    // so we check the pid is actually alive. kill(pid, 0) tests existence
    // without signalling; EPERM means it exists but is not ours, which still
    // counts as attached.
    std::ifstream f(marker);
    long long pid = 0;
    if (!(f >> pid) || pid <= 0) {
        // Unreadable or malformed marker: refuse rather than guess. A gc that
        // prunes while a daemon holds objects open is the exact trap gc.hpp
        // warns about, and refusing costs only a manual file removal.
        return SFS_ERR(MalformedObject, "unreadable mount marker; refusing to gc",
                       marker.string());
    }

    if (::kill(static_cast<::pid_t>(pid), 0) == 0) return true;
    if (errno == EPERM) return true;

    return false;  // stale marker, no live daemon
}

Result<GcReport> gc(core::IBlockStore& blocks, CommitStore& commits, ManifestStore& manifests,
                    RefStore& refs, const core::RepoPaths& paths, const GcOptions& opts) {
    GcReport report;

    // gc MUST take the exclusive lock (gc.hpp). Everything below either
    // unlinks objects or rewrites them, and a concurrent commit would be
    // walking the same refs.
    auto lock = RepoLock::acquire(paths.lock(), LockMode::Exclusive);
    if (!lock) return std::unexpected(lock.error());

    auto attached = mount_attached(paths);
    if (!attached) return std::unexpected(attached.error());
    if (*attached) {
        return SFS_ERR(RepositoryLocked,
                       "a mount daemon is attached; unmount before running gc");
    }

    report.temp_files_removed =
        sweep_temp_dir(paths.tmp(), opts.dry_run, report.bytes_reclaimed);
    report.temp_files_removed +=
        sweep_temp_dir(paths.incoming(), opts.dry_run, report.bytes_reclaimed);

    if (!opts.prune) return report;

    // Reachability is THE definition of live (dag.hpp), shared with push so
    // the two can never disagree. HEAD is included alongside the branch refs:
    // a detached HEAD is not in refs/heads and its commit must not be pruned
    // out from under the user.
    auto heads = refs.list_heads();
    if (!heads) return std::unexpected(heads.error());

    std::vector<Oid> tips;
    tips.reserve(heads->size() + 1);
    for (const auto& h : *heads) tips.push_back(h.target);

    if (auto head = refs.read_head(); head && head->is_detached()) {
        tips.push_back(*head->detached);
    }

    // No refs at all means nothing is reachable. Pruning here would empty a
    // freshly-initialised repository of any objects written before the first
    // ref update, so we refuse to treat "no tips" as "everything is garbage".
    if (tips.empty()) return report;

    auto live = reachable_objects(commits, manifests, tips);
    if (!live) return std::unexpected(live.error());

    // Pruning needs the concrete store: unlink() and list_all() are
    // LooseStore/BlockStore operations, deliberately absent from the
    // IBlockStore seam (nothing on the read path may delete anything).
    auto* store = dynamic_cast<BlockStore*>(&blocks);
    if (!store) {
        return SFS_ERR(NotImplemented, "prune requires a concrete BlockStore");
    }

    auto all = store->list_all();
    if (!all) return std::unexpected(all.error());

    for (const auto& oid : *all) {
        ++report.objects_scanned;
        if (live->count(oid)) continue;

        std::error_code ec;
        auto sz = std::filesystem::file_size(paths.object_path(oid), ec);
        if (ec) sz = 0;

        if (!opts.dry_run) {
            // Unlink through the object path rather than the store: the
            // object is already known unreachable, and BlockStore exposes no
            // public unlink (only LooseStore does, for gc --pack).
            std::error_code rm_ec;
            std::filesystem::remove(paths.object_path(oid), rm_ec);
            if (rm_ec) continue;
        }
        ++report.objects_pruned;
        report.bytes_reclaimed += sz;
    }

    if (opts.pack) {
        // Packfiles are additive and explicitly optional (ADR 0006); the
        // loose store is the reference implementation and journal.cpp's pack
        // recovery is likewise unimplemented. Reporting zero packed is
        // honest; silently succeeding would not be.
        return SFS_ERR(NotImplemented, "gc --pack is not implemented in this build");
    }

    return report;
}

}  // namespace sfs::store
