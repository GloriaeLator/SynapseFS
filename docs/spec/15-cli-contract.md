# SPEC 15 - CLI contract

## 1. Exit codes

```cpp
enum ExitCode : int {
    Ok             = 0,
    Failure        = 1,   // generic
    Usage          = 2,   // CLI11 argument-parsing errors
    NotARepository = 3,
    Integrity      = 4,   // hash mismatch, tamper detection, violated invariant
    Conflict       = 5,   // merge needs a resolution
    Locked         = 6,   // repo lock held, or a mount is attached during gc
    NotImplemented = 7,
    Network        = 8,   // declared, never returned by any command today
};
```

`exit_code_for(const core::Error&)` maps a propagated `core::Error` to one
of these via `Error::exit_code()`. Both codes are reachable, though the
first is easy to misread as dead:

- `NotImplemented` (7) - no `apps/sfs/cmd/*.cpp` file writes the literal
  `ExitCode::NotImplemented`/`7`, which reads like the `apps/sfs/CMakeLists.txt`
  comment's promise ("a command that is wired but unimplemented exits 7") is
  unmet. `Error::exit_code()` maps `ErrKind::NotImplemented` to `7` (`modules/core/src/error.cpp`)
- `Network` (8)  - `run_push`/`run_pull`
  (`apps/sfs/cmd/push.cpp`/`pull.cpp`) now return `ExitCode::Network` for `net::push`/`net::pull` on failure (see §9).

## 2. `sfs init [<directory>]`

Creates `.synapsefs/{objects,tmp,refs/heads,journal}`, writes a default
`RepoConfig`, sets `HEAD` to `ref: refs/heads/main`. Fails (`Failure`) if
`.synapsefs` already exists at the target.

## 3. `sfs commit <file.safetensors> -m <message> [--author <name>] [--topology <config.json>]`

`--author` defaults to `$USER`. `--topology` defaults to `<file's
directory>/config.json` if that path exists, else no topology is used. See
[`architecture.md`](../architecture.md#sfs-commit) for the full pipeline,
including the silent fallback to full-tensor storage on any alignment
failure. Fails `NotARepository` if not in a repo, `Failure` if HEAD is
detached ("create a branch before committing") or the file doesn't exist.

## 4. `sfs checkout <revision> [-o/--output <path>]`

`<revision>` is a branch name, a full or abbreviated oid, or a
`refs/heads/<name>` path. If it names an existing branch, HEAD becomes
symbolic to that branch (a pure switch - pre-2.23 git semantics, no
"detected as branch" ambiguity). Otherwise HEAD becomes detached at the
resolved oid. With no `--output`, nothing is materialized - only the HEAD
move happens, and the command prints "Switched to branch '<name>'" or "HEAD
is now at `<abbrev>`". With `--output <path>`, streams
`codec::reconstruct_file()` into that path via `stio::StWriter`
(atomic-rename-on-success, refuses to leave a partial/mismatched file - see
[`storage_format.md`](../storage_format.md)).

## 5. `sfs branch [<name>] [<start-point>] [-d/--delete <name>] [-f/--force]`

No `<name>` -> lists every branch, current one marked `* `. `<name>` with no
`<start-point>` -> creates at the current HEAD tip (`Failure` if HEAD has no
commits yet). `-d` requires `<name>` (`Usage` otherwise) and deletes via
`refs.delete_branch`. **`branch` never switches the current branch** -
that is `checkout <branch>`, matching pre-2.23 git semantics, and is
explicit in the source's own header comment.

## 6. `sfs log [<revision>] [-n/--max-count <N>]`

Defaults to HEAD, `-n -1` (unlimited). Walks commit ancestry
(`store::walk_commits`), printing `commit <oid>`, `Merge: <parents...>` for
merge commits, `Author:`, `Date:`, then the message. If HEAD has no
commits, prints "fatal: your current branch has no commits yet" - this
specific message is used both as a genuine `Failure` (when resolving HEAD's
tip fails) and, separately, as a plain informational print that still exits
`Ok` (when a valid but empty history is walked) - check the exit code, not
just the message, if scripting against this.

## 7. `sfs verify [<revision>] [--full] [--repair]`

Standalone - does not require checkout or mount. No `<revision>` -> checks
every branch head. `--repair` runs journal recovery first
(`Journal::recover`), *then* verification. Default mode checks object
existence, ancestor invariants, and chain-depth consistency; `--full`
additionally re-hashes every chunk of every referenced object. Prints
`commits walked:` / `objects checked:`, then either `verify: OK` (exit
`Ok`) or one `INTEGRITY: <kind> <oid> [group=][chunk=]: <detail>` line per
finding followed by `verify: FAILED (<N> finding(s))` (exit **`Integrity`
= 4**). Findings always name the failing object's oid. The `chunk=` field
is populated for a `--full` chunk-payload mismatch (`ErrKind::HashMismatch`
on the whole-object check): `store::verify()` re-derives the exact failing
chunk index by re-walking the object through `IBlockStore::read_range()`
(which verifies chunk-by-chunk internally) and reading the index back out
of that call's own `ChunkDigestMismatch` error, without needing a new
`IBlockStore` method. It's left unset for a non-chunk-shaped failure (a
missing object, a malformed container) where no single chunk is
responsible.

## 8. `sfs merge <branch> [--ours | --theirs] [-m <message>]`

`--ours` and `--theirs` together is `Usage` (2). Requires a non-detached
HEAD. A fast-forward prints "Already up to date." (if merging into self)
or just moves the ref with no new commit. A true three-way merge with
unresolved conflicts (default strategy, no `--ours`/`--theirs`) prints each
conflicting group's `ours=`/`theirs=` block ids, writes nothing, and exits
**`Conflict` = 5**. A clean or auto-resolved merge writes a two-parent
commit and prints "Merge made by three-way strategy." then
`[<ref> <abbrev>] <message>`. See [`storage_format.md`](../storage_format.md#merge)
for the resolution rules - this is a real per-tensor-group merge.

## 9. `sfs push <remote_url>` / `sfs pull <remote_url>`

`<remote_url>` must be a bare `IP:PORT` string; the `--help` text used to
say `http://ip:port` (wrong, and a scheme-prefixed URL used to crash the
process uncaught) and now says `ip:port` - `connect_to_remote` also
tolerates a copy-pasted `scheme://` prefix defensively and never throws on
malformed input either way (see [SPEC 14](14-wire-protocol.md#2-public-api)).
**Both commands now propagate the underlying transfer's success/failure**:
a connection failure, a dropped transfer, or (for `pull` specifically - see
SPEC 14 §4) a received object that fails its integrity check exits
**`Network` = 8** with a message on stderr, instead of always exiting `Ok`.
`push` cannot detect a receiver-side integrity failure over this protocol
(no ack channel back to the sender - SPEC 14 §4); treat a pushed-to repo as
unverified until it runs its own `sfs verify --full`.

## 10. `sfs serve [-p/--port <port>]`

Defaults `-p` to `9418`, matching `RepoConfig::listen`'s documented default
(it used to have no default at all, silently binding an OS-assigned
ephemeral port - see [SPEC 14](14-wire-protocol.md#5-server)). Blocks
forever in a single-connection-at-a-time accept loop; the command does not
exit under normal operation.

## 11. `sfs mount <revision> <mountpoint> [-f/--foreground] [--debug]` (built only if `SFS_BUILD_MOUNT`)

Resolves the revision to a commit/manifest, starts the FUSE3 low-level
daemon, registers a PID marker (`.synapsefs/mount-daemon.pid`) so `gc` can
refuse while attached, prints "Mounted `<abbrev>` at `<mountpoint>`
(read-only)", then blocks running the FUSE session loop until unmounted.
Mount is always read-only - `open()` rejects any non-`O_RDONLY` flag with
`EROFS`.

## 12. `sfs unmount <mountpoint>` (built only if `SFS_BUILD_MOUNT`)

Shells out to `fusermount3 -u <mountpoint>`.
