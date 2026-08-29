#pragma once
/// \file interfaces.hpp
/// THE SEAMS. Frozen the same way the on-disk formats are: three teams compile
/// against this file, so changing a signature needs the same sign-off as
/// changing a spec.
///
/// What is virtual and what is not is argued in
/// docs/adr/0010-virtual-dispatch-vs-templates.md. Short version: a call that
/// happens once per frame or less can be virtual for free; a call that happens
/// once per unit or per byte must not be.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/oid.hpp>
#include <synapsefs/core/tensor.hpp>

/// The manifest lives in `format`, which depends on `core` — so it is
/// forward-declared here rather than included. IObjectSource is the seam, and a
/// seam that pulled in its collaborator's headers would not be one.
namespace sfs::format {
struct Manifest;
}  // namespace sfs::format

namespace sfs::core {

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// The seam between `codec` (knows permutations, not repositories) and `store`
/// (the reverse).
class IBlockStore {
public:
    virtual ~IBlockStore() = default;

    /// Idempotent: writing identical bytes twice writes nothing the second time.
    [[nodiscard]] virtual Result<Oid> put(ObjectKind, std::span<const std::byte> payload) = 0;

    /// Whole object, kind-checked, fully verified.
    [[nodiscard]] virtual Result<std::vector<std::byte>> get(const Oid&, ObjectKind) = 0;

    /// FAST PATH. Copies [offset, offset + out.size()) into `out`, verifying
    /// only the chunks it touches. This is what the mount calls on a fault, so
    /// it writes into a caller-owned span: nothing on the fault path allocates.
    /// Returns bytes written; short only at end of object.
    [[nodiscard]] virtual Result<std::size_t> read_range(const Oid&, ObjectKind,
                                                         std::uint64_t offset,
                                                         std::span<std::byte> out) = 0;

    /// SLOW PATH. Every chunk, then the object digest against its own address.
    /// Use when ingesting from a peer, in `verify`, and after a crash.
    [[nodiscard]] virtual Status verify_block(const Oid&, ObjectKind) = 0;

    [[nodiscard]] virtual Result<bool>          contains(const Oid&) const = 0;
    [[nodiscard]] virtual Result<std::uint64_t> size_of(const Oid&) const = 0;
    [[nodiscard]] virtual Result<ObjectKind>    kind_of(const Oid&) const = 0;
};

/// Everything codec::read_range needs from a repository, without codec
/// depending on store.
class IObjectSource {
public:
    virtual ~IObjectSource() = default;
    [[nodiscard]] virtual Result<const format::Manifest*> manifest_for(
        const Oid& commit) = 0;
    /// The ancestor invariant: a delta's base commit must be an ancestor of the
    /// commit containing it. docs/spec/10-object-model.md §4.3.
    [[nodiscard]] virtual Result<bool> is_ancestor(const Oid& maybe_ancestor,
                                                   const Oid& of) = 0;
};

// ---------------------------------------------------------------------------
// Checkpoint input
// ---------------------------------------------------------------------------

/// Lazy access to a .safetensors file. The ONLY way align and stio read
/// weights, so that the out-of-core path is the only path (ADR 0008).
class ITensorSource {
public:
    virtual ~ITensorSource() = default;

    /// Verbatim [8-byte LE length][JSON header], including trailing padding.
    /// Stored as its own block; regenerating it is how you get a file that is
    /// four bytes wrong with every tensor bit-identical.
    [[nodiscard]] virtual std::span<const std::byte> header_bytes() const = 0;

    /// In BUFFER order, which is neither key order nor topology order.
    [[nodiscard]] virtual std::span<const BufferEntry> buffer_layout() const = 0;

    [[nodiscard]] virtual const TensorMeta* meta(std::string_view name) const = 0;
    [[nodiscard]] virtual std::uint64_t total_bytes() const = 0;

    /// Read `count` output units starting at `first`. Never loads the whole
    /// tensor. Bytes are as they appear in the file.
    [[nodiscard]] virtual Result<std::size_t> read_units(std::string_view name,
                                                         std::uint64_t first,
                                                         std::uint64_t count,
                                                         std::span<std::byte> out) = 0;
};

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

struct LapResult {
    std::vector<std::uint32_t> assignment;      ///< a permutation of [0, n)
    double        cost_raw        = 0.0;
    double        cost_normalized = 0.0;
    bool          exact           = false;
    std::uint32_t iterations      = 0;
};

/// Exact Jonker-Volgenant below the measured crossover, greedy + local 2-swap
/// above it. The crossover and its accuracy cost are measured, not guessed —
/// see docs/benchmarks.md. The solver never sees a checkpoint, which makes
/// test_lap.cpp a pure algorithmic test with planted optima.
class ILapSolver {
public:
    virtual ~ILapSolver() = default;
    /// `cost` is row-major n x n. Minimises total assignment cost.
    [[nodiscard]] virtual Result<LapResult> solve(std::span<const float> cost,
                                                  std::uint32_t n) = 0;
    [[nodiscard]] virtual std::string_view name() const = 0;
};

// ---------------------------------------------------------------------------
// Networking
// ---------------------------------------------------------------------------

enum class FrameType : std::uint8_t;   // net/frame.hpp

struct WireFrame {
    FrameType                type{};
    std::vector<std::byte>   payload;
};

/// So that test_havewant.cpp runs the real negotiation with no sockets, and
/// sync_interrupt cuts a transfer at a chosen byte offset deterministically
/// rather than by racing a kill.
class ITransport {
public:
    virtual ~ITransport() = default;
    [[nodiscard]] virtual Status send(FrameType, std::span<const std::byte> payload) = 0;
    [[nodiscard]] virtual Result<WireFrame> recv(std::chrono::milliseconds timeout) = 0;
    virtual void close() = 0;
};

}  // namespace sfs::core
