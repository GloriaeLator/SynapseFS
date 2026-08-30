/// Unit tests for the gc/mount interlock.
///
/// gc MUST NOT run while a mount daemon is attached: the daemon holds objects
/// open by id, and an unlinked-but-open file is a trap that fires on the next
/// remount (gc.hpp). mount::Daemon::register_with_repo writes the marker;
/// store::mount_attached reads it. The two are two sides of one contract and
/// live in different modules, so this test pins the file format down from the
/// store side.

#include <catch2/catch_test_macros.hpp>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <synapsefs/store/gc.hpp>

using namespace sfs;

namespace {

/// A bare .synapsefs directory. gc's marker check runs before any store is
/// opened, so a full repository is not needed to exercise it.
class TempDot {
public:
    TempDot() {
        paths_.root = std::filesystem::temp_directory_path() /
                      ("sfs-gc-test-" + std::to_string(::getpid()) + "-" +
                       std::to_string(counter_++));
        std::filesystem::create_directories(paths_.dot());
    }
    ~TempDot() {
        std::error_code ec;
        std::filesystem::remove_all(paths_.root, ec);
    }
    TempDot(const TempDot&) = delete;
    TempDot& operator=(const TempDot&) = delete;

    [[nodiscard]] const core::RepoPaths& paths() const noexcept { return paths_; }

    void write_marker(const std::string& contents) const {
        std::ofstream f(paths_.dot() / "mount-daemon.pid");
        f << contents;
    }

private:
    core::RepoPaths paths_;
    static inline int counter_ = 0;
};

}  // namespace

TEST_CASE("no marker means no daemon attached", "[gc]") {
    TempDot dot;
    auto attached = store::mount_attached(dot.paths());
    REQUIRE(attached.has_value());
    REQUIRE_FALSE(*attached);
}

TEST_CASE("a marker naming a live process means attached", "[gc]") {
    TempDot dot;
    // Our own pid is the one process we can be certain is alive.
    dot.write_marker(std::to_string(::getpid()) + "\n/mnt/whatever\n");

    auto attached = store::mount_attached(dot.paths());
    REQUIRE(attached.has_value());
    REQUIRE(*attached);
}

TEST_CASE("a stale marker does not block gc forever", "[gc]") {
    TempDot dot;
    // A daemon killed with -9 never removes its marker. If a stale marker
    // counted as attached, one crashed mount would make gc refuse
    // permanently, which is worse than the trap it is guarding against.
    // PID 0x7FFFFFFF is above any Linux pid_max and cannot be running.
    dot.write_marker("2147483647\n/mnt/whatever\n");

    auto attached = store::mount_attached(dot.paths());
    REQUIRE(attached.has_value());
    REQUIRE_FALSE(*attached);
}

TEST_CASE("an unreadable marker refuses rather than guessing", "[gc]") {
    TempDot dot;
    // We cannot tell whether a daemon is attached. Pruning on a guess risks
    // the exact corruption gc.hpp warns about; refusing costs one manual
    // file removal.
    SECTION("empty") {
        dot.write_marker("");
        REQUIRE_FALSE(store::mount_attached(dot.paths()).has_value());
    }
    SECTION("not a number") {
        dot.write_marker("not-a-pid\n");
        REQUIRE_FALSE(store::mount_attached(dot.paths()).has_value());
    }
    SECTION("nonsensical pid") {
        dot.write_marker("-5\n");
        REQUIRE_FALSE(store::mount_attached(dot.paths()).has_value());
    }
}
