#pragma once
/// \file packfile.hpp
/// Packfiles. OPTIONAL — on the cut list, and cutting it costs a performance
/// number, not a feature (docs/adr/0006).
///
/// Objects keep their chunk digests inside a pack, so read_range behaves
/// identically whether an object is loose or packed and the mount does not know
/// the difference. Pack names are themselves content-addressed (the digest of
/// the sorted list of contained oids), which makes them safe to name in a sync.

#include <filesystem>
#include <span>
#include <vector>

#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/format/object.hpp>

namespace sfs::store {

struct PackEntry {
    core::Oid        oid;
    core::ObjectKind kind{};
    std::uint64_t    offset = 0;   ///< into the .sfspack
    std::uint64_t    length = 0;
};

/// Read-only view of one pack plus its index. Immutable once written.
class Packfile {
public:
    [[nodiscard]] static core::Result<Packfile> open(const std::filesystem::path& pack_path);
    ~Packfile();
    Packfile(Packfile&&) noexcept;
    Packfile& operator=(Packfile&&) noexcept;
    Packfile(const Packfile&) = delete;
    Packfile& operator=(const Packfile&) = delete;

    [[nodiscard]] bool contains(const core::Oid&) const;
    [[nodiscard]] core::Result<std::size_t> read_range(const core::Oid&, core::ObjectKind,
                                                       std::uint64_t offset,
                                                       std::span<std::byte> out);
    [[nodiscard]] core::Status verify_block(const core::Oid&, core::ObjectKind);
    [[nodiscard]] std::span<const PackEntry> entries() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Packfile();
};

/// Writes pack + index atomically. The old pack is unlinked only after both
/// new files are fsynced and in place.
class PackWriter {
public:
    explicit PackWriter(std::filesystem::path pack_dir, std::filesystem::path tmp_dir);
    [[nodiscard]] core::Status add(const core::Oid&, core::ObjectKind,
                                   std::span<const std::byte> object_file_bytes);
    [[nodiscard]] core::Result<std::filesystem::path> finish();

private:
    std::filesystem::path  pack_dir_;
    std::filesystem::path  tmp_dir_;
    std::vector<PackEntry> entries_;
};

}  // namespace sfs::store
