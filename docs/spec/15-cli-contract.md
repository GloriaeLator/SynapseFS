# SPEC 15 — CLI contract

**Status:** normative

`sfs` is the single binary. Twelve subcommands, stable flags, machine-readable
output where it helps, and exit codes that mean something.

This is a contract because three teams implement against it and because the
evaluator types these commands. A command that does the right thing with a
different flag name still fails the demo.

---

## 1. Invocation

```
sfs [--repo <path>] [-v|--verbose]... [-q|--quiet] [--json] [--no-color] <command> [args]
```

| Global flag | Meaning |
|---|---|
| `--repo <path>` | Repository root. Default: search upward from `$PWD` for `.synapsefs/`. |
| `-v`, `--verbose` | Repeatable. Once = info, twice = debug, three times = trace. |
| `-q`, `--quiet` | Errors only. |
| `--json` | Machine-readable output on stdout. Human text goes to stderr. |
| `--no-color` | Also implied when stdout is not a TTY, and by `NO_COLOR`. |
| `--version` | `sfs 0.1.0 (<git sha>, blake3 <ver>, zstd <ver>, fuse <ver>)` and exit 0. |

`sfs --help` MUST list every command. Progress goes to **stderr**, so that
`sfs log --json | jq` works.

---

## 2. Commands

### `sfs init [<path>]`
Creates `.synapsefs/` with `format`, an empty `objects/`, `refs/heads/`, and
`HEAD` pointing at `refs/heads/main`. Idempotent-ish: refuses a non-empty
`.synapsefs/` rather than merging into it.

### `sfs commit <checkpoint.safetensors> -m <msg> [--config <config.json>] [--author <a>] [--no-delta]`
Parses the checkpoint, builds or reuses a topology, aligns each group against
the parent commit, writes objects, writes the manifest and commit, moves the
current branch. `--no-delta` forces every group to `mode: full` — the escape
hatch when alignment is suspected and the thing you want is a correct commit.

Prints the new commit identifier. With `--json`, `{"commit": "b3:…", "bytes_written": N, "groups": {"full": n, "delta": m}}`.

### `sfs checkout <ref|oid> [--out <path>] [-b <branch>]`
Two jobs, as in pre-2.23 git and as the PS specifies:

- `checkout <branch-name>` switches the current branch;
- `checkout <ref> --out <path>` materialises the checkpoint file.

`-b <branch>` creates the branch and switches to it. Reconstruction is a loop
over `read_range` (SPEC 12 §6) — the same primitive the mount uses.

### `sfs branch [<name>] [--list] [-d <name>]`
Creates or lists. **Does not switch** — that is `checkout`. `-d` deletes a ref;
refuses if the branch is not reachable from another ref unless `--force`.

### `sfs log [<ref>] [--graph] [--max-count N] [--json]`
Walks `parents` from the given ref (default `HEAD`).

### `sfs verify [<ref>] [--full] [--repair]`
**Must work standalone**, with no checkout and no mount. This is explicit in
the PS and it is a graded metric.

- default: walk the DAG from every ref, check every object's identifier, check
  the ancestor invariant, check chunk digests for objects it reads;
- `--full`: additionally re-verify every chunk of every reachable object;
- `--repair`: replay or roll back a journal record left by a crash (SPEC 11
  §3.3), then re-verify.

Output names the failing object *and* the failing chunk. Exit code 4 on
detected corruption — distinct from exit 1, so a script can tell "corrupt" from
"failed to run".

### `sfs merge <branch> [--ours|--theirs] [-m <msg>]`
Fast-forward when one tip is an ancestor of the other. Otherwise three-way per
tensor group against the merge base: changed on one side takes that side;
changed on **both** sides is a conflict and `merge` **refuses**, listing the
conflicting groups, until `--ours` or `--theirs` picks a side wholesale.

Averaging conflicting weights would be a defensible research idea and an
indefensible version-control one: a VCS that silently produces an artifact
neither author wrote is broken. `docs/tradeoffs.md` argues this at length
because it is the kind of thing the Q&A asks about.

### `sfs push <remote-url> [<branch>] [--force]` · `sfs pull <remote-url> [<branch>]`
SPEC 14. `push` sends objects then a compare-and-swap ref update. `pull`
fetches and fast-forwards, refusing a divergent history and telling the user to
`merge`.

### `sfs serve [--listen host:port] [--repo <path>] [--read-only]`
Starts the sync listener. Address, port and config **must be documented in the
README** — a listed deliverable. Default `127.0.0.1:9418`.

### `sfs mount <ref> <mountpoint> [--foreground] [--cache-bytes N] [--allow-other]`
Mounts a read-only view. `<mountpoint>/<file.name>` is the checkpoint from that
commit. `--foreground` is required under a sanitizer build and is what the demo
uses so `strace` output is visible.

### `sfs unmount <mountpoint>`
Unmounts, waiting for in-flight reads. Falls back to `fusermount3 -u`.

### `sfs gc [--pack] [--prune] [--dry-run]`
Repacks loose objects and removes unreachable ones. Refuses while a mount
daemon is attached (SPEC 11 §6).

---

## 3. Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Generic failure |
| 2 | Usage error — bad flags, missing argument (what CLI11 produces) |
| 3 | Not a repository, or repository format unsupported |
| 4 | **Integrity failure** — a hash mismatch, a tamper detection, a violated invariant |
| 5 | Conflict — merge requires a resolution |
| 6 | Locked — another process holds the write lock |
| 7 | Not implemented in this build |
| 8 | Network failure |

Code 4 is deliberately its own value. "The data is wrong" and "the program is
wrong" are different events for anyone scripting this, and the crash and tamper
harnesses distinguish them.

A command that is wired but unimplemented MUST exit 7 with a one-line message
naming what is missing. Silently succeeding is worse than failing, and during a
build week an honest exit 7 is what keeps the integration script meaningful.

---

## 4. Output conventions

- Identifiers abbreviate to 12 hex characters in human output, never in
  `--json`.
- Sizes are human-readable (`1.4 GiB`) in human output, exact byte integers in
  `--json`.
- Every long operation prints progress to stderr with a byte counter, and
  prints nothing when stderr is not a TTY.
- Errors are one line: `sfs: <command>: <what failed>: <path or oid>`, with
  detail on following indented lines.

---

## 5. Test hooks

| Assertion | Test |
|---|---|
| `--help` lists every command; each has help text | `tests/test_end_to_end.cpp` |
| Every unimplemented command exits 7 with a message | same |
| `verify` on a tampered repo exits 4 and names the chunk | `tests/tamper.cpp` |
| `merge` with both sides changed exits 5 and lists groups | `modules/store/tests/test_merge_logic.cpp` |
| `--json` output parses for `log`, `commit`, `verify` | `tests/e2e.py` |
