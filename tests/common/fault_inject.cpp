/// \file fault_inject.cpp
/// Implements tests/common/fault_inject.hpp.
///
/// Both classes are wrappers around the ABSTRACT core::IBlockStore interface,
/// not the concrete store::LooseStore/BlockStore implementation, so they can
/// only inject faults at the granularity that interface exposes:
///
///   FaultInjectingStore re-verifies "at rest" corruption itself using the
///   same primitive real objects are addressed by (core::compute_oid), so a
///   detection here is a genuine hash-mismatch computation, not a canned
///   failure — but it cannot reach into the real loose-object file on disk
///   (IBlockStore exposes no such thing), so the corruption lives in the
///   wrapper, applied to whatever bytes `inner` legitimately hands back.
///
///   CrashingStore can only fail before or after one call to `put()`, since
///   put() is atomic-or-nothing from IBlockStore's point of view. The real
///   write/rename/fsync sequence `put()` performs internally
///   (docs/storage_format.md, atomic_write) has THREE distinguishable crash
///   windows; from outside this interface only TWO are observable (before
///   the call took effect, or after) — so `AfterWriteBeforeRename` and
///   `AfterRenameBeforeDirFsync` necessarily behave identically here. Telling
///   them apart for real needs a fault hook inside store::BlockStore's own
///   atomic_write() call, which is future work (see docs/known-gaps.md),
///   not something an interface-level wrapper can fake honestly.

#include "fault_inject.hpp"

#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include <synapsefs/core/oid.hpp>

namespace sfs::test {

using core::ObjectKind;
using core::Oid;
using core::Result;
using core::Status;

namespace {

/// Flip bit 0 of every byte in `offsets` that falls within `buf`. Offsets
/// past the end of a short object are silently ignored — a test asking to
/// corrupt "the last byte of every object kind" should not itself need to
/// know each kind's exact size.
void apply_offsets(std::vector<std::byte>& buf, const std::vector<std::uint64_t>& offsets) {
    for (auto off : offsets) {
        if (off < buf.size()) {
            buf[static_cast<std::size_t>(off)] ^= std::byte{0x01};
        }
    }
}

void apply_offsets_in_range(std::span<std::byte> out, std::uint64_t range_start,
                            const std::vector<std::uint64_t>& offsets) {
    for (auto off : offsets) {
        if (off >= range_start && off - range_start < out.size()) {
            out[static_cast<std::size_t>(off - range_start)] ^= std::byte{0x01};
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// FaultInjectingStore
// ---------------------------------------------------------------------------

struct FaultInjectingStore::Impl {
    core::IBlockStore& inner;
    std::mutex mu;
    std::map<std::string, std::vector<std::uint64_t>> read_faults;  ///< oid.to_string() -> offsets
    std::map<std::string, std::vector<std::uint64_t>> rest_faults;

    explicit Impl(core::IBlockStore& s) : inner(s) {}
};

FaultInjectingStore::FaultInjectingStore(core::IBlockStore& inner)
    : impl_(std::make_shared<Impl>(inner)) {}

void FaultInjectingStore::corrupt_on_read(const Oid& oid, std::uint64_t offset) {
    std::lock_guard lock(impl_->mu);
    impl_->read_faults[oid.to_string()].push_back(offset);
}

void FaultInjectingStore::corrupt_at_rest(const Oid& oid, std::uint64_t offset) {
    std::lock_guard lock(impl_->mu);
    impl_->rest_faults[oid.to_string()].push_back(offset);
}

void FaultInjectingStore::clear() {
    std::lock_guard lock(impl_->mu);
    impl_->read_faults.clear();
    impl_->rest_faults.clear();
}

Result<Oid> FaultInjectingStore::put(ObjectKind kind, std::span<const std::byte> payload) {
    return impl_->inner.put(kind, payload);
}

Result<std::vector<std::byte>> FaultInjectingStore::get(const Oid& oid, ObjectKind kind) {
    auto bytes = impl_->inner.get(oid, kind);
    if (!bytes) return bytes;
    auto buf = std::move(*bytes);

    std::vector<std::uint64_t> rest_offsets;
    std::vector<std::uint64_t> read_offsets;
    {
        std::lock_guard lock(impl_->mu);
        if (auto it = impl_->rest_faults.find(oid.to_string()); it != impl_->rest_faults.end())
            rest_offsets = it->second;
        if (auto it = impl_->read_faults.find(oid.to_string()); it != impl_->read_faults.end())
            read_offsets = it->second;
    }

    if (!rest_offsets.empty()) {
        // "At rest": the object's stored bytes are wrong. A real LooseStore's
        // get() re-hashes the whole payload against the object's own address
        // before ever returning it (docs/storage_format.md), so the honest
        // simulation is: apply the corruption, recompute the SAME framed
        // digest real objects are addressed by, and fail exactly the way the
        // real store would if its file were actually corrupted like this.
        auto corrupted = buf;
        apply_offsets(corrupted, rest_offsets);
        if (core::compute_oid(kind, corrupted) != oid) {
            return SFS_ERR(HashMismatch, "FaultInjectingStore: at-rest corruption detected",
                          oid.to_string());
        }
        // A flipped bit that happens not to change the digest (astronomically
        // unlikely for BLAKE3, but not impossible for a degenerate offset) —
        // fall through and hand back the corrupted-but-coincidentally-valid
        // bytes, same as a real store would.
        buf = std::move(corrupted);
    }

    if (!read_offsets.empty()) {
        // Read-path corruption: applied AFTER the store's own verification
        // already passed, modeling a bit flip between the block store and its
        // caller. This one is deliberately NOT self-detected here — it tests
        // whether something downstream (a manifest witness hash, a frame
        // digest inside the payload) catches it, not the block store itself.
        apply_offsets(buf, read_offsets);
    }

    return buf;
}

Result<std::size_t> FaultInjectingStore::read_range(const Oid& oid, ObjectKind kind,
                                                     std::uint64_t offset,
                                                     std::span<std::byte> out) {
    auto n = impl_->inner.read_range(oid, kind, offset, out);
    if (!n) return n;

    std::vector<std::uint64_t> rest_offsets;
    std::vector<std::uint64_t> read_offsets;
    {
        std::lock_guard lock(impl_->mu);
        if (auto it = impl_->rest_faults.find(oid.to_string()); it != impl_->rest_faults.end())
            rest_offsets = it->second;
        if (auto it = impl_->read_faults.find(oid.to_string()); it != impl_->read_faults.end())
            read_offsets = it->second;
    }

    // read_range's whole reason to exist (docs/storage_format.md) is that it
    // verifies only the chunks the read touches — so unlike get(), an at-rest
    // fault whose offset falls OUTSIDE [offset, offset+n) here should NOT be
    // caught by this call, matching the real store's chunk-local guarantee.
    // Only recompute-and-compare when the corrupted offset is actually in the
    // touched range, and even then we can only compare the touched slice, not
    // the whole object's digest — so this falls back to the same "apply, then
    // let the caller's own chunk-digest re-check (format::verify_chunk, if it
    // uses one) decide" behaviour as an in-range at-rest fault. In practice
    // tamper.cpp should prefer get()/verify_block() for at-rest cases and
    // reserve read_range() for corrupt_on_read.
    apply_offsets_in_range(out.subspan(0, *n), offset, rest_offsets);
    apply_offsets_in_range(out.subspan(0, *n), offset, read_offsets);
    return n;
}

Status FaultInjectingStore::verify_block(const Oid& oid, ObjectKind kind) {
    if (auto st = impl_->inner.verify_block(oid, kind); !st) return st;

    std::vector<std::uint64_t> rest_offsets;
    {
        std::lock_guard lock(impl_->mu);
        if (auto it = impl_->rest_faults.find(oid.to_string()); it != impl_->rest_faults.end())
            rest_offsets = it->second;
    }
    if (rest_offsets.empty()) return {};

    // The underlying store's own verify_block() passed (its real file is
    // untouched — see this file's header comment on what a wrapper can and
    // cannot reach), so independently re-derive what verify_block() SHOULD
    // report for the corruption this test configured, using the real fetch
    // path (which re-verifies against the true, uncorrupted stored bytes)
    // before applying the simulated corruption and checking it the same way
    // get() does above.
    auto bytes = impl_->inner.get(oid, kind);
    if (!bytes) return std::unexpected(bytes.error());
    auto corrupted = std::move(*bytes);
    apply_offsets(corrupted, rest_offsets);
    if (core::compute_oid(kind, corrupted) != oid) {
        return SFS_ERR(HashMismatch, "FaultInjectingStore: at-rest corruption detected",
                      oid.to_string());
    }
    return {};
}

Result<bool> FaultInjectingStore::contains(const Oid& oid) const {
    return impl_->inner.contains(oid);
}

Result<std::uint64_t> FaultInjectingStore::size_of(const Oid& oid) const {
    return impl_->inner.size_of(oid);
}

Result<ObjectKind> FaultInjectingStore::kind_of(const Oid& oid) const {
    return impl_->inner.kind_of(oid);
}

// ---------------------------------------------------------------------------
// CrashingStore
// ---------------------------------------------------------------------------

struct CrashingStore::Impl {
    core::IBlockStore& inner;
    std::uint64_t fail_on_nth_put;
    When when;
    std::uint64_t put_count = 0;
    std::mutex mu;

    Impl(core::IBlockStore& s, std::uint64_t n, When w) : inner(s), fail_on_nth_put(n), when(w) {}
};

CrashingStore::CrashingStore(core::IBlockStore& inner, std::uint64_t fail_on_nth_put, When when)
    : impl_(std::make_shared<Impl>(inner, fail_on_nth_put, when)) {}

Result<Oid> CrashingStore::put(ObjectKind kind, std::span<const std::byte> payload) {
    std::uint64_t n;
    {
        std::lock_guard lock(impl_->mu);
        n = ++impl_->put_count;
    }
    if (n != impl_->fail_on_nth_put) return impl_->inner.put(kind, payload);

    if (impl_->when == When::BeforeWrite) {
        // Died before the payload ever left this call — the underlying store
        // never sees it, so no object exists at all for whatever oid it would
        // have had.
        return SFS_ERR(Io, "CrashingStore: simulated crash before write");
    }

    // AfterWriteBeforeRename / AfterRenameBeforeDirFsync: from outside
    // IBlockStore's atomic put(), both look identical — the call either took
    // effect or it didn't, and if we're here it did (see this file's header
    // comment for why finer-grained injection needs a hook inside
    // store::BlockStore itself). So: really perform the write (the object
    // becomes durable, exactly as a crash after the real rename would leave
    // it), then report failure anyway — modeling the caller never learning
    // the oid succeeded (e.g. the process died before the ref/journal update
    // that would have recorded it). A repository in this state should be
    // exactly what docs/known-gaps.md's "Working as designed" note
    // describes: an unreferenced but perfectly valid object, wasted disk, not
    // corruption.
    auto real = impl_->inner.put(kind, payload);
    if (!real) return real;  // the underlying store itself failed; report that, not our fake crash
    return SFS_ERR(Io, "CrashingStore: simulated crash after write, before caller observed success");
}

Result<std::vector<std::byte>> CrashingStore::get(const Oid& oid, ObjectKind kind) {
    return impl_->inner.get(oid, kind);
}

Result<std::size_t> CrashingStore::read_range(const Oid& oid, ObjectKind kind,
                                              std::uint64_t offset, std::span<std::byte> out) {
    return impl_->inner.read_range(oid, kind, offset, out);
}

Status CrashingStore::verify_block(const Oid& oid, ObjectKind kind) {
    return impl_->inner.verify_block(oid, kind);
}

Result<bool> CrashingStore::contains(const Oid& oid) const { return impl_->inner.contains(oid); }

Result<std::uint64_t> CrashingStore::size_of(const Oid& oid) const {
    return impl_->inner.size_of(oid);
}

Result<ObjectKind> CrashingStore::kind_of(const Oid& oid) const {
    return impl_->inner.kind_of(oid);
}

}  // namespace sfs::test
