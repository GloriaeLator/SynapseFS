# Ownership

Who owns what, and therefore who to ask. Eight people, three teams, four days.

Two rules behind this table:

- **Every person owns a named component end to end** and can explain it without
  the author of the neighbouring component in the room. The rubric treats a
  feature nobody can defend as not implemented, and the Y26s get a separate
  implementation-specific Q&A.
- **Nobody gets glue work.** If a role's description is "wiring", that is a bug
  in this table.

---

## Roles

Fill in the names on Day 1 and keep this file current — it is the first thing a
reviewer reads and the first thing a stranger needs.

| Role | Name | Owns | Modules |
|---|---|---|---|
| **A1** | _TBD_ | Topology parsing, permutation groups, cost function, LAP solver, propagation, confidence and `alignable: false` | `align/` |
| **A2** | _TBD_ (Y26) | Diff artifact format, residual codec, frames, compression, SIMD kernels | `codec/`, `format/residual_hdr` |
| **A3** | _TBD_ | Buffer-layout builder, `read_range`, reconstruction, byte-exactness harness → **joins Mount from Day 3** | `stio/`, `codec/reconstruct` |
| **V1** | _TBD_ | Block store, atomic I/O, locks, chunk digests, range reads, packfiles, chain resolution, snapshot policy | `store/`, `util/atomic_io` |
| **V2** | _TBD_ (Y26) | Commit / manifest / refs / branch / merge / `verify`, the tamper suite | `store/`, `format/` |
| **V3** | _TBD_ | CLI wiring and exit codes, then networking (`serve` / `push` / `pull`) | `apps/sfs/`, `net/` |
| **M1** | _TBD_ | FUSE daemon, open/read/mmap path, concurrency and single-flight fill | `mount/` |
| **M2** | _TBD_ (Y26) | Interval table, LRU frame cache, RSS, benchmarks | `mount/`, `bench/` |

Effective headcount: Align 3 → 2 from Day 3; Vault 3 throughout; Mount 2 → 3
from Day 3. A3 moving to Mount is not a reassignment — `read_range` *is* the
seam between reconstruction and the daemon, so it moves with the person who
wrote it.

---

## Who to ask

| Question | Ask |
|---|---|
| "Why is my permutation the wrong length?" | A1 — blocking factors, SPEC 13 |
| "Why doesn't my delta reconstruct?" | A2 for the artifact, A3 for `read_range` |
| "Where does this byte in the file come from?" | A3 — buffer layout |
| "Why did my write not survive the crash test?" | V1 — SPEC 11 §3 |
| "Why did `verify` exit 4?" | V2 — it names the object and chunk; read that first |
| "What exit code should this be?" | V3 — SPEC 15 §3 |
| "Why is `mmap` returning zeros?" | M1 — `FOPEN_DIRECT_IO` is almost certainly set |
| "Why is peak RSS over budget?" | M2 — frame cache accounting |
| "Can we change this field?" | All three team leads. The formats are frozen. |

---

## Documentation ownership

15% of the grade, and the rubric penalises explanations the authors cannot
defend. Each document is written by the person who built the thing.

| Doc | Owner |
|---|---|
| `README.md` | V3 |
| `docs/architecture.md` | A3 + V1 |
| `docs/storage_format.md` | V1 |
| `docs/alignment_algorithm.md` | A1 |
| `docs/tradeoffs.md` | A2 + V2 |
| `docs/threat_model.md` | V2 |
| `docs/benchmarks.md` | M2 |
| `docs/spec/*` | The owner of the module that writes the format |
| `docs/adr/*` | Whoever made the decision |

**Three sentences per person per day**, on the piece you built, in your own
words, committed that day. It is not optional and it is not busywork: it is the
raw material for both the docs and the Q&A, and it is the difference between a
defensible README and a penalised one.

---

## Changing a frozen format

The specs in `docs/spec/` and the interfaces in
`modules/core/include/synapsefs/core/interfaces.hpp` are frozen. To change one:

1. Sign-off from all three team leads.
2. A **single commit** that updates the spec, the golden objects in
   `tests/golden/`, and every affected test.
3. A note in `docs/tradeoffs.md` saying what changed and why.

Three teams read these objects. A schema change on Day 3 costs a day.
