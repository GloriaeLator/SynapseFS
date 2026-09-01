#pragma once
/// \file mount_test_common.hpp
/// Minimal in-memory IBlockStore + a "one header, one or more Full/Raw
/// groups" manifest builder, so mount's unit tests can drive
/// SynapseFs::create() / read() without a real repository on disk.
///
/// Deliberately does not exercise Delta groups or codec::read_range's chain
/// walk — that is modules/codec/tests/test_byte_identity.cpp's job. This
/// double only needs to be good enough to make read_range() on a Raw block
/// return real bytes, since that's all the mount layer (interval lookup,
/// frame cache, copy) actually depends on.

#include <cstdint>
#include <cstring>
#include <map>
#include <span>
#include <string>
#include <vector>

#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/core/oid.hpp>
#include <synapsefs/format/manifest.hpp>

namespace sfs::test {

/// Same shape as modules/codec/tests/test_byte_identity.cpp's FakeBlockStore.
class FakeBlockStore final : public core::IBlockStore {
public:
    core::Result<core::Oid> put(core::ObjectKind kind, std::span<const std::byte> payload) override {
        auto oid = core::compute_oid(kind, payload);
        objs_[oid] = std::vector<std::byte>(payload.begin(), payload.end());
        kinds_[oid] = kind;
        return oid;
    }
    core::Result<std::vector<std::byte>> get(const core::Oid& oid, core::ObjectKind) override {
        auto it = objs_.find(oid);
        if (it == objs_.end()) return SFS_ERR(ObjectNotFound, "no such object", oid.to_string());
        return it->second;
    }
    core::Result<std::size_t> read_range(const core::Oid& oid, core::ObjectKind,
                                         std::uint64_t offset, std::span<std::byte> out) override {
        auto it = objs_.find(oid);
        if (it == objs_.end()) return SFS_ERR(ObjectNotFound, "no such object", oid.to_string());
        const auto& bytes = it->second;
        if (offset >= bytes.size()) return std::size_t{0};
        const std::size_t n = std::min<std::size_t>(out.size(), bytes.size() - offset);
        std::memcpy(out.data(), bytes.data() + offset, n);
        return n;
    }
    core::Status verify_block(const core::Oid& oid, core::ObjectKind) override {
        // Mirrors real digest checking closely enough for tamper-style
        // tests: a byte flipped via corrupt() must be detectable somehow by
        // callers that compare against an independently-known expectation.
        // The mount's own read path relies on codec's chunk digests, which
        // this double does not model; verify_block here is a no-op success,
        // matching FakeBlockStore in test_byte_identity.cpp.
        if (objs_.find(oid) == objs_.end()) return SFS_ERR(ObjectNotFound, "no such object");
        return {};
    }
    core::Result<bool> contains(const core::Oid& oid) const override { return objs_.count(oid) > 0; }
    core::Result<std::uint64_t> size_of(const core::Oid& oid) const override {
        auto it = objs_.find(oid);
        if (it == objs_.end()) return SFS_ERR(ObjectNotFound, "no such object");
        return it->second.size();
    }
    core::Result<core::ObjectKind> kind_of(const core::Oid& oid) const override {
        auto it = kinds_.find(oid);
        if (it == kinds_.end()) return SFS_ERR(ObjectNotFound, "no such object");
        return it->second;
    }

    /// Test-only: flip a bit so a later read observes corruption.
    void corrupt(const core::Oid& oid, std::size_t byte_index) {
        objs_.at(oid)[byte_index] ^= std::byte(0xFF);
    }

private:
    std::map<core::Oid, std::vector<std::byte>>  objs_;
    std::map<core::Oid, core::ObjectKind>         kinds_;
};

/// Never actually walked (no Delta groups in this fixture), but ReadCtx and
/// codec::read_range require a live IObjectSource pointer.
class NullObjectSource final : public core::IObjectSource {
public:
    core::Result<const format::Manifest*> manifest_for(const core::Oid&) override {
        return SFS_ERR(ObjectNotFound, "fixture has no delta history");
    }
    core::Result<bool> is_ancestor(const core::Oid&, const core::Oid&) override { return true; }
};

/// One synthetic ".safetensors"-shaped file: a verbatim header block plus N
/// named groups, each a Full/Raw block of caller-chosen bytes, laid out
/// contiguously in the order given — i.e. exactly the layout
/// IntervalTable::build documents (header, then buffer entries in buffer
/// order, cursor == running offset).
struct GroupSpec {
    std::string             name;
    std::vector<std::byte>  bytes;
};

struct BuiltFixture {
    FakeBlockStore    store;
    format::Manifest  manifest;
    std::vector<std::vector<std::byte>> group_bytes;  ///< same order as `groups` in, for expected-value checks
};

inline std::vector<std::byte> make_bytes(std::size_t n, std::uint8_t seed) {
    std::vector<std::byte> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = std::byte(static_cast<std::uint8_t>(seed + i * 7 + i / 251));
    return v;
}

/// Builds a manifest with a header block of `header_bytes.size()` bytes,
/// followed by one buffer entry per group (each its own tensor AND its own
/// permutation group, one-to-one, which is all IntervalTable cares about).
inline BuiltFixture build_fixture(const std::vector<std::byte>& header_bytes,
                                  const std::vector<GroupSpec>& groups) {
    BuiltFixture fx;

    auto header_oid = fx.store.put(core::ObjectKind::Header, header_bytes);

    fx.manifest.file.name = "model.safetensors";
    fx.manifest.file.header_block = *header_oid;

    std::uint64_t cursor = header_bytes.size();
    for (const auto& g : groups) {
        auto block_oid = fx.store.put(core::ObjectKind::Raw, g.bytes);

        format::GroupEntry ge;
        ge.mode = format::GroupMode::Full;
        ge.block = *block_oid;
        ge.chain_depth = 0;
        fx.manifest.groups[g.name] = ge;

        core::BufferEntry be;
        be.tensor = g.name;
        be.off    = cursor;
        be.nbytes = g.bytes.size();
        be.group  = g.name;
        fx.manifest.buffer.push_back(be);

        cursor += g.bytes.size();
        fx.group_bytes.push_back(g.bytes);
    }

    fx.manifest.file.total_bytes = cursor;
    return fx;
}

/// Concatenation of the header and every group's bytes, in layout order --
/// i.e. what a correct sequential read of the whole file must reproduce.
inline std::vector<std::byte> flatten(const std::vector<std::byte>& header,
                                      const std::vector<GroupSpec>& groups) {
    std::vector<std::byte> out = header;
    for (const auto& g : groups) out.insert(out.end(), g.bytes.begin(), g.bytes.end());
    return out;
}

}  // namespace sfs::test
