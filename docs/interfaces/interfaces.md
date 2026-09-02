# Cross-module interfaces

The seams, exactly as declared in
`modules/core/include/synapsefs/core/interfaces.hpp` - all pure-virtual,
zero implementation in `core` itself.

## `IBlockStore`

```cpp
Result<Oid>    put(ObjectKind, span<const byte> payload);          // idempotent
Result<vector<byte>> get(Oid, ObjectKind);                          // fully verified
Result<size_t> read_range(Oid, ObjectKind, uint64_t offset, span<byte> out);
Status         verify_block(Oid, ObjectKind);
bool           contains(Oid) const;
Result<uint64_t> size_of(Oid) const;
Result<ObjectKind> kind_of(Oid) const;
```

`read_range` is the fast path - caller-owned buffer, no allocation, only
the chunks touched by `[offset, offset+out.size())` are re-hashed. This is
what the FUSE mount calls on a page fault and what `checkout`/`verify`
(default mode) use for referenced-object handling. `verify_block` is the
slow path used by `sfs verify --full`; as implemented
(`store::LooseStore`) it is a full-object re-hash (equivalent to calling
`get()` and discarding the payload), not a chunk-by-chunk walk, even though
the on-disk chunk table exists.

Implemented by `store::BlockStore`/`store::LooseStore`. Deliberately has no
delete operation - nothing on the read path can remove an object; `gc
--prune` reaches around this interface (via `dynamic_cast<BlockStore*>`) to
call `std::filesystem::remove` directly.

## `IObjectSource`

```cpp
Result<const format::Manifest*> manifest_for(Oid commit);
Result<bool> is_ancestor(Oid maybe_ancestor, Oid of);
```

`format::Manifest` is only forward-declared in `core/interfaces.hpp` - by
design, since `format` depends on `core`, not the reverse.
`is_ancestor` is what enforces the "ancestor invariant": a Delta group's
recorded base commit must actually be an ancestor of the commit whose
manifest references it. Implemented by `store`.

## `ITensorSource`

```cpp
span<const byte>          header_bytes() const;              // verbatim [8-byte len][JSON]
span<const core::BufferEntry> buffer_layout() const;          // buffer order, not key/topology order
Result<const TensorMeta*> meta(string_view name) const;
uint64_t                  total_bytes() const;
Result<size_t>             read_units(string_view name, uint64_t first,
                                       uint64_t count, span<byte> out);
```

`read_units` never loads a whole tensor; it reads exactly the requested
unit range. This is the seam that keeps alignment out-of-core (ADR 0008) -
nothing that talks to a tensor through this interface has to hold the whole
thing in memory. Implemented by `stio::StSource` for a real `.safetensors`
file on disk, and by internal `ReconstructedTensorSource`/
`ColumnPermutingSource` types (in `store`) for reading a *reconstructed*
parent checkpoint during `commit`, without ever materializing it as a file.

Note: `stio::StSource::read_units` is currently an **axis-0-only**
implementation (`row_stride = nbytes / shape[0]`) - the axis-agnostic
primitive the interface implies (`stio::row_iter.hpp`'s `UnitReader`) is
declared but has no implementation anywhere in the tree and no callers.
There is currently no working path for reading tensor units along any axis
other than axis 0. See [`known-gaps.md`](../known-gaps.md).

## `ILapSolver`

```cpp
struct LapResult { vector<uint32_t> assignment; double cost_raw;
                    double cost_normalized; bool exact; uint32_t iterations; };
Result<LapResult> solve(span<const float> cost_row_major, uint32_t n);
string_view name() const;
```

Three concrete implementations exist in `modules/align`: an exact
Jonker–Volgenant solver, a greedy+2-swap heuristic, and (for very large
groups) a Bertsekas parallel-auction path that bypasses this interface
entirely and operates on `torch::Tensor`s directly. See
[`alignment_algorithm.md`](../alignment_algorithm.md).

## `ITransport`

```cpp
Status send(span<const byte>);
Result<WireFrame> recv(chrono::milliseconds timeout);
Status close();
```

with `WireFrame { FrameType type; vector<byte> payload; }` (`FrameType` is
only forward-declared in `core`, defined in `net`). **This interface is
declared but has no implementation and no caller anywhere in the
repository.** `modules/net`'s actual push/pull/serve implementation talks
to raw BSD sockets directly and never constructs a `WireFrame` or an
`ITransport`. See [`spec/14-wire-protocol.md`](../spec/14-wire-protocol.md).
