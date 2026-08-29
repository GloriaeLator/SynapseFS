#pragma once
/// \file block_store.hpp
/// The composite block store: loose objects, optionally over packfiles.
/// Implements core::IBlockStore.
///
/// Two read paths, and the distinction is the whole reason tamper detection and
/// mmap throughput are not a trade-off here:
///   read_range()   verifies only the 64 KiB chunks it touches  -> 183.9 MB/s
///   verify_block() verifies everything                         ->   0.6 MB/s
/// docs/storage_format.md §3.

#include <filesystem>
#include <memory>

#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/format/object.hpp>

namespace sfs::store {

using core::Oid;
using core::ObjectKind;
using core::Result;
using core::Status;

class BlockStore final : public core::IBlockStore {
public:
    [[nodiscard]] static Result<std::unique_ptr<BlockStore>> open(const core::RepoPaths&,
                                                                  const core::RepoConfig&);
    ~BlockStore() override;

    [[nodiscard]] Result<Oid> put(ObjectKind, std::span<const std::byte>) override;
    [[nodiscard]] Result<std::vector<std::byte>> get(const Oid&, ObjectKind) override;
    [[nodiscard]] Result<std::size_t> read_range(const Oid&, ObjectKind, std::uint64_t offset,
                                                 std::span<std::byte> out) override;
    [[nodiscard]] Status verify_block(const Oid&, ObjectKind) override;
    [[nodiscard]] Result<bool> contains(const Oid&) const override;
    [[nodiscard]] Result<std::uint64_t> size_of(const Oid&) const override;
    [[nodiscard]] Result<ObjectKind> kind_of(const Oid&) const override;

    /// Streamed write for objects too large to hold in memory (a `full` group
    /// of a 7B checkpoint). Chunk digests accumulate as bytes arrive.
    class Writer {
    public:
        virtual ~Writer() = default;
        [[nodiscard]] virtual Status write(std::span<const std::byte>) = 0;
        /// Atomically publishes the object and returns its oid.
        [[nodiscard]] virtual Result<Oid> commit() = 0;
        virtual void abort() noexcept = 0;
    };
    [[nodiscard]] Result<std::unique_ptr<Writer>> begin_put(ObjectKind,
                                                            std::uint64_t payload_len);

    /// Enumerate every object present. Used by gc and by `verify --full`.
    [[nodiscard]] Result<std::vector<Oid>> list_all() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    BlockStore();
};

}  // namespace sfs::store
