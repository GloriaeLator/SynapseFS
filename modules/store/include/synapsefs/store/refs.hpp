#pragma once
/// \file refs.hpp
/// Branch refs and HEAD. docs/spec/11-repo-layout.md §4.
///
/// A ref file is exactly one line: "b3:<64 hex>\n", written with atomic_write.
/// Updates are compare-and-swap, which is what makes concurrent commits safe
/// under the repository lock and what makes `pull` refuse a non-fast-forward
/// instead of silently discarding history.
///
/// `branch` creates and lists only; switching is `checkout <branch-name>`,
/// pre-2.23 git semantics, as the PS specifies.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>
#include <synapsefs/core/repo_config.hpp>

namespace sfs::store {

using core::Oid;
using core::Result;
using core::Status;

struct Ref {
    std::string name;    ///< "refs/heads/main"
    Oid         target;
};

/// HEAD is either a symbolic ref or a detached oid.
struct Head {
    std::optional<std::string> symbolic;   ///< "refs/heads/main"
    std::optional<Oid>         detached;

    [[nodiscard]] bool is_detached() const noexcept { return detached.has_value(); }
};

class RefStore {
public:
    explicit RefStore(core::RepoPaths);

    [[nodiscard]] Result<Oid>  resolve(std::string_view name_or_oid) const;
    [[nodiscard]] Result<std::vector<Ref>> list_heads() const;
    [[nodiscard]] Result<Head> read_head() const;
    [[nodiscard]] Status       set_head_symbolic(std::string_view ref_name);
    [[nodiscard]] Status       set_head_detached(const Oid&);

    /// Compare-and-swap. `expected` empty means "must not currently exist".
    /// Returns ErrKind::RefRaceLost if the current value differs.
    [[nodiscard]] Status update(std::string_view ref_name, std::optional<Oid> expected,
                                const Oid& desired);

    [[nodiscard]] Status create_branch(std::string_view name, const Oid& at);
    /// Refuses unless the branch is reachable from another ref, or force.
    [[nodiscard]] Status delete_branch(std::string_view name, bool force);

    /// Resolves "HEAD", "main", "refs/heads/main", "b3:...", and abbreviations
    /// in human input only (never inside a stored object).
    [[nodiscard]] Result<Oid> rev_parse(std::string_view) const;

private:
    core::RepoPaths paths_;
};

}  // namespace sfs::store
