/// `sfs unmount <mountpoint>`. See mount.cpp for the SFS_WITH_MOUNT guard
/// rationale.
#ifdef SFS_WITH_MOUNT

#include <CLI/CLI.hpp>

#include <iostream>

#include <synapsefs/mount/daemon.hpp>

#include "../exitcode.hpp"

namespace sfs::app::cmd {

namespace {

int run_unmount(const std::string& mountpoint) {
    if (auto st = mount::unmount(mountpoint); !st) {
        std::cerr << "error: " << st.error().to_string() << "\n";
        return exit_code_for(st.error());
    }
    std::cout << "Unmounted " << mountpoint << "\n";
    return ExitCode::Ok;
}

}  // namespace

void register_unmount(CLI::App& app, int& exit_code) {
    static std::string mountpoint;
    auto* c = app.add_subcommand("unmount", "Unmount a synapsefs FUSE mount");
    c->add_option("mountpoint", mountpoint, "Directory to unmount")->required();
    c->callback([&exit_code] { exit_code = run_unmount(mountpoint); });
}

}  // namespace sfs::app::cmd

#endif  // SFS_WITH_MOUNT
