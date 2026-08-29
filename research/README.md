# research/

Experiments that produced numbers in the ADRs and in `docs/benchmarks.md`.
Not part of the build; not needed to run `sfs`.

| File | Question it answers |
|---|---|
| `lap_bench.py` | Where does the exact LAP solver stop being acceptable, and what does the greedy fallback cost in accuracy? (ADR 0004, benchmarks §1) |
| `cost_ablation.py` | Which cost function recovers planted permutations most reliably: inner product, cosine, with/without incoming slices, with/without folded norm statistics? |
| `notebooks/` | Scratch. Nothing here is load-bearing. |

Anything from here that ends up justifying a decision gets written into an ADR
or into `docs/tradeoffs.md` **with the number**. A result that lives only in a
notebook is a result nobody can defend in the Q&A.
