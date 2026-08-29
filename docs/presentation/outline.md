# Presentation outline

Ten minutes maximum, at least two presenters, heavy Q&A graded on whether you
can defend the code. A feature nobody can answer a question about is treated as
not implemented — and a worse solution whose authors can explain it scores above
a better one whose authors cannot.

## Shape

| Time | Content | Presenter |
|---|---|---|
| 0:00–1:00 | The two problems, with **our own measured numbers**: % of bytes churned by one fine-tune step, % churned by a pure permutation | _TBD_ |
| 1:00–3:30 | Alignment: permutation groups, LAP, propagation, out-of-core. **One diagram** of a permutation crossing a residual block | _TBD_ |
| 3:30–5:30 | Storage: the object graph, chunk digests, why verification granularity equals read granularity. The crash-harness iteration counter | _TBD_ |
| 5:30–7:30 | **Live demo**: mount, `load_file()` from Python, `strace` showing the daemon never writes a file | _TBD_ |
| 7:30–9:00 | The five numbers, and two trade-offs we measured and **rejected** | _TBD_ |
| 9:00–10:00 | What is out of scope and why | _TBD_ |

## The slides that are ours rather than generic

- **The four-byte failure.** Every tensor bit-identical, the file still wrong.
  It is a one-slide argument for why the manifest stores the header verbatim.
- **1.3 seconds for one page fault.** The whole-layer versus frames table.
- **0.6 MB/s versus 183.9 MB/s**, and the chunked version *names* the corrupt
  chunk. Verification granularity = read granularity is the strongest
  original-thinking claim in Module 2.
- **The crash-harness counter.** "We ran N iterations of kill-9-at-a-random-point
  and every one recovered or refused" is a different claim from "we handle
  crashes".
- **`alignable: false`.** Most teams will not have it.

## Rehearsal

Two full run-throughs, with the non-presenters asking hostile questions. Anything
that does not land goes on the asker's study list for that evening.

Y26s get a separate implementation-specific Q&A. Rehearse that separately, on
implementation detail rather than concepts.

## Demo safety

- Pre-generate fixtures. Never generate live.
- Have the repo already initialised and one commit in.
- Terminal font large enough to read from the back.
- A recorded fallback of the mount demo, in case `/dev/fuse` is unavailable on
  the room's machine.
