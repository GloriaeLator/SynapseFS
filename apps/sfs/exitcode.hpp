#pragma once
/// \file exitcode.hpp
/// Process exit codes. docs/spec/15-cli-contract.md §3.
///
/// Code 4 is deliberately its own value: "the data is wrong" and "the program
/// is wrong" are different events for anyone scripting this, and the crash and
/// tamper harnesses both depend on telling them apart.

#include <synapsefs/core/error.hpp>

namespace sfs::app {

enum ExitCode : int {
    Ok             = 0,
    Failure        = 1,
    Usage          = 2,   ///< what CLI11 produces
    NotARepository = 3,
    Integrity      = 4,   ///< a hash mismatch, a tamper detection, a violated invariant
    Conflict       = 5,   ///< merge needs a resolution
    Locked         = 6,
    NotImplemented = 7,
    Network        = 8,
};

[[nodiscard]] inline int exit_code_for(const core::Error& e) noexcept {
    return e.exit_code();
}

}  // namespace sfs::app
