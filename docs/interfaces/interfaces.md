# Cross-module interfaces

These are the seams. They live in `modules/core/include/synapsefs/core/interfaces.hpp`,
depend on nothing but `util` and `core`'s own value types, and are what lets a
module be built and tested before its collaborators exist.

**They are frozen the same way the on-disk formats are.** Changing a signature
here needs the same sign-off, because three teams compile against it.

Which things are virtual and which are not is argued in
[ADR 0010](../adr/0010-virtual-dispatch-vs-templates.md). The short version: if
a call happens once per frame or less, a virtual call is free; if it happens
once per unit or per byte, it must not be virtual.

---

## `IBlockStore`

The seam between `codec` (which knows about permutations and nothing about
repositories) and `store` (which is the reverse).

```cpp
class IBlockStore {
public:
    virtual ~IBlockStore() = default;

    /// Store `payload` under kind `k`. Returns its oid. Idempotent: writing
    /// identical bytes twice writes nothing the second time.
    virtual Result<Oid> put(ObjectKind k, std::span<const std::byte> payload) = 0;

    /// Whole object, kind-checked. Verifies the full digest.
    virtual Result<std::vector<std::byte>> get(const Oid&, ObjectKind) = 0;

    /// FAST PATH. Copies [offset, offset+out.size()) into `out`, verifying only
    /// the 64 KiB chunks it touches. This is what the mount calls on a fault.
    /// Returns bytes written (short only at end of object).
    virtual Result<std::size_t> read_range(const Oid&, ObjectKind,
                                           std::uint64_t offset,
                                           std::span<std::byte> out) = 0;

    /// SLOW PATH. Full verification: every chunk, then the object digest
    /// against its own address. Use when ingesting from a peer, in `verify`,
    /// and after a crash.
    virtual Status verify_block(const Oid&, ObjectKind) = 0;

    virtual Result<bool>          contains(const Oid&) const = 0;
    virtual Result<std::uint64_t> size_of(const Oid&)  const = 0;
    virtual Result<ObjectKind>    kind_of(const Oid&)  const = 0;
};
```

`read_range` writes into a caller-owned span rather than returning a vector, so
that the fault path allocates nothing. That is the one place we let a
performance concern shape an interface, and it is deliberate.

Implementations: `LooseStore`, `PackedStore`, `CompositeStore` (loose over
pack), `MemStore` (tests), `FaultInjectingStore` (tamper and crash matrices).

---

## `IObjectSource`

Everything `read_range` in `codec` needs from a repository, without `codec`
depending on `store`.

```cpp
struct ReadCtx {
    IBlockStore*    blocks;
    const Manifest* manifest;
    IObjectSource*  history;     // to resolve a delta's base commit
    FrameCache*     cache;       // nullable; the mount supplies one
    std::uint32_t   max_depth;
};

class IObjectSource {
public:
    virtual ~IObjectSource() = default;
    virtual Result<const Manifest*> manifest_for(const Oid& commit) = 0;
    virtual Result<bool> is_ancestor(const Oid& maybe_ancestor, const Oid& of) = 0;
};
```

---

## `ITensorSource`

Lazy access to a `.safetensors` file. The only way `align` and `stio` read
weights, so that the out-of-core path is the *only* path
([ADR 0008](../adr/0008-out-of-core-streaming.md)).

```cpp
class ITensorSource {
public:
    virtual ~ITensorSource() = default;

    virtual std::span<const std::byte> header_bytes() const = 0;   // verbatim, with padding
    virtual std::span<const TensorEntry> buffer_layout() const = 0; // buffer order
    virtual std::uint64_t total_bytes() const = 0;

    /// Read `count` output units starting at `first` from tensor `name`.
    /// Never loads the whole tensor. Bytes are as they appear in the file.
    virtual Result<std::size_t> read_units(std::string_view name,
                                           std::uint64_t first, std::uint64_t count,
                                           std::span<std::byte> out) = 0;
};
```

---

## `ILapSolver`

Chosen at runtime by problem size: exact Jonker–Volgenant below the measured
crossover, greedy + local 2-swap above it
([ADR 0004](../adr/0004-weight-matching-vs-activation-vs-ot.md)).

```cpp
struct LapResult {
    std::vector<std::uint32_t> assignment;  // a permutation of [0, n)
    double cost_raw = 0.0;
    double cost_normalized = 0.0;
    bool   exact = false;
    std::uint32_t iterations = 0;
};

class ILapSolver {
public:
    virtual ~ILapSolver() = default;
    /// `cost` is row-major n x n. Minimises total assignment cost.
    virtual Result<LapResult> solve(std::span<const float> cost, std::uint32_t n) = 0;
    virtual std::string_view name() const = 0;
};
```

The solver never sees a checkpoint. That makes `test_lap.cpp` a pure
algorithmic test with planted optima and no fixtures.

---

## `ITransport`

So that `test_havewant.cpp` runs the real negotiation with no sockets.

```cpp
class ITransport {
public:
    virtual ~ITransport() = default;
    virtual Status send(FrameType, std::span<const std::byte> payload) = 0;
    virtual Result<Frame> recv(std::chrono::milliseconds timeout) = 0;
    virtual void close() = 0;
};
```

Implementations: `TcpTransport`, `PipeTransport` (in-process, tests),
`FlakyTransport` (drops the connection at a chosen byte offset — this is how
`sync_interrupt` is tested deterministically rather than by racing a `kill`).

---

## `IResidualKernel` — *not* an interface

Worth stating explicitly, because someone will propose it.

The XOR/zigzag inner loop is dispatched **once per frame** through a plain
function pointer selected from CPUID at startup
([ADR 0011](../adr/0011-simd-dispatch-strategy.md)):

```cpp
using XorApplyFn = void (*)(std::byte* dst, const std::byte* base,
                            const std::byte* resid, std::size_t n);
XorApplyFn xor_apply_dispatch() noexcept;   // codec/kernels/dispatch.cpp
```

No virtual call, no object, nothing per-unit. `SFS_FORCE_ISA=scalar|avx2|avx512`
overrides the selection for benchmarking and bisection.

---

## Test doubles

Every interface above has a double in `tests/common/`. Two of them are load
bearing rather than convenient:

| Double | What it makes possible |
|---|---|
| `FaultInjectingStore` | Flip a byte in any object at any chain position, on read or at rest. This *is* `tests/tamper.cpp`. |
| `CrashingStore` | Fail the Nth write, or die between the rename and the directory fsync. This *is* `tests/crash_matrix.cpp`. |
| `FlakyTransport` | Cut a transfer at byte N deterministically. This *is* `tests/sync_interrupt.cpp`. |

If an interface cannot be doubled cheaply, that is a signal the seam is in the
wrong place.
