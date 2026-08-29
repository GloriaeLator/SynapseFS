#pragma once
/// \file fault_inject.hpp
/// The test doubles that make the tamper and crash matrices possible.
///
/// These are not conveniences. Deterministic fault injection beats racing a
/// real kill -9 because a failing case is reproducible: the double proves each
/// stage is handled, and the real-kill harness finds the stage nobody thought
/// of. We run both.

#include <cstdint>
#include <functional>
#include <memory>

#include <synapsefs/core/interfaces.hpp>

namespace sfs::test {

/// Wraps a store and corrupts bytes on read, or at rest. Used by tamper.cpp to
/// flip one byte in each object kind at each position in a chain and assert
/// that every case is detected AND NAMED.
class FaultInjectingStore final : public core::IBlockStore {
public:
    explicit FaultInjectingStore(core::IBlockStore& inner);

    /// Flip bit 0 of byte `offset` of `oid` whenever it is read.
    void corrupt_on_read(const core::Oid&, std::uint64_t offset);
    /// Corrupt the stored bytes, so verify_block must catch it too.
    void corrupt_at_rest(const core::Oid&, std::uint64_t offset);
    void clear();

    [[nodiscard]] core::Result<core::Oid> put(core::ObjectKind,
                                              std::span<const std::byte>) override;
    [[nodiscard]] core::Result<std::vector<std::byte>> get(const core::Oid&,
                                                           core::ObjectKind) override;
    [[nodiscard]] core::Result<std::size_t> read_range(const core::Oid&, core::ObjectKind,
                                                       std::uint64_t,
                                                       std::span<std::byte>) override;
    [[nodiscard]] core::Status verify_block(const core::Oid&, core::ObjectKind) override;
    [[nodiscard]] core::Result<bool> contains(const core::Oid&) const override;
    [[nodiscard]] core::Result<std::uint64_t> size_of(const core::Oid&) const override;
    [[nodiscard]] core::Result<core::ObjectKind> kind_of(const core::Oid&) const override;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

/// Fails the Nth write, or dies between the rename and the parent-directory
/// fsync — the exact window that atomic_write exists to close.
class CrashingStore final : public core::IBlockStore {
public:
    enum class When { BeforeWrite, AfterWriteBeforeRename, AfterRenameBeforeDirFsync };

    CrashingStore(core::IBlockStore& inner, std::uint64_t fail_on_nth_put, When);

    [[nodiscard]] core::Result<core::Oid> put(core::ObjectKind,
                                              std::span<const std::byte>) override;
    [[nodiscard]] core::Result<std::vector<std::byte>> get(const core::Oid&,
                                                           core::ObjectKind) override;
    [[nodiscard]] core::Result<std::size_t> read_range(const core::Oid&, core::ObjectKind,
                                                       std::uint64_t,
                                                       std::span<std::byte>) override;
    [[nodiscard]] core::Status verify_block(const core::Oid&, core::ObjectKind) override;
    [[nodiscard]] core::Result<bool> contains(const core::Oid&) const override;
    [[nodiscard]] core::Result<std::uint64_t> size_of(const core::Oid&) const override;
    [[nodiscard]] core::Result<core::ObjectKind> kind_of(const core::Oid&) const override;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace sfs::test
