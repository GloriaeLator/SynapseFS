#pragma once
/// \file commit_store.hpp
/// Reading and writing commit objects, and the ordering rule that makes a
/// crashed commit harmless.
///
///   write and fsync every object  ->  fsync the objects directory
///                                 ->  compare-and-swap the ref
///
/// A crash before the ref update leaves unreferenced objects. Nothing points at
/// them, verify does not walk them, gc collects them: the failure mode of a
/// crashed commit is wasted disk, not a broken repository.

#include <memory>
#include <optional>

#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/format/commit.hpp>
#include <synapsefs/store/refs.hpp>

namespace sfs::store {

class CommitStore {
public:
    CommitStore(core::IBlockStore& blocks, RefStore& refs);

    [[nodiscard]] Result<format::Commit> read(const Oid&) const;
    [[nodiscard]] Result<Oid>            write(const format::Commit&);

    /// Write the commit object and move `ref_name` to it, in that order, as a
    /// compare-and-swap against `expected_tip`.
    [[nodiscard]] Result<Oid> commit_and_advance(const format::Commit&,
                                                 std::string_view ref_name,
                                                 std::optional<Oid> expected_tip);

    /// Cached parse — `log` and `verify` walk the same commits repeatedly.
    [[nodiscard]] Result<const format::Commit*> get_cached(const Oid&);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace sfs::store
