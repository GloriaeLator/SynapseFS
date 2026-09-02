# SPEC 14 - Wire protocol

## 1. What this is, precisely

A bespoke, unauthenticated, unencrypted, plaintext, line-delimited TCP
protocol that synchronizes the raw files under `.synapsefs/` by comparing
**file size** (or, for ref/HEAD files, raw content), not by comparing
object hashes or walking the commit graph. `synapse_sync.cpp` calls raw BSD
sockets directly.

There is no protocol version field and no magic bytes. Cryptographic
verification of transferred bytes based out of how this protocol moves: a received loose object (anything landing under
`.synapsefs/objects/`) is checked against its own claimed address before
being committed into place (§4). Refs/HEAD/journal files are still synced
by raw content/size equality only - see [`threat_model.md`](../threat_model.md).

## 2. Public API

```cpp
namespace sfs::net {
    bool push(const std::string& remote_url);   // remote_url = "IP:PORT"
    bool pull(const std::string& remote_url);   // remote_url = "IP:PORT"
    void serve(int port);                       // blocking accept() loop, never returns
}
```

`connect_to_remote(url)` splits `url` at the **first** `:` and expects a
bare `IP:PORT` string. The help text reads `ip:port ...
NOT a URL, no http:// or other scheme prefix`, and `connect_to_remote()`
itself defensively strips a leading `"<scheme>://"` if one is present (so a
copy-pasted `http://ip:port` still works rather than crashing), parses the
port with a `safe_stoi` helper that returns `std::nullopt` instead of
throwing on garbage, validates the parsed port is in `(0, 65535]`, and
checks the `socket()`/`inet_pton()` return values before using them. A
malformed remote string is now a clean connection failure (`push`/`pull`
return `false`, see §6), never an uncaught exception.

## 3. Handshake and framing

Every exchange is plain ASCII, newline-terminated, read one byte at a time
by a line reader (`read_line()` - functionally correct, not efficient).
There is no binary frame header anywhere in this protocol.

1. Client connects and sends the literal line `PUSH` or `PULL`.
2. **Inventory exchange** (`send_inventory`/`receive_inventory`): a
   `<decimal length>\n` line, followed by exactly that many raw bytes of a
   `nlohmann::json` object mapping local `.synapsefs` file paths to either
   their byte size, or - for any path containing `/ref` or `HEAD` - their
   literal file contents (used to detect ref changes by string equality
   rather than size).
3. **Request** (receiver -> sender): `REQ\n<filepath>\n<decimal offset>\n`.
4. **File response** (sender -> receiver): `FILE\n<filepath>\n<decimal
   total_size>\n<decimal offset>\n` immediately followed by
   `total_size - offset` raw bytes, streamed in 8192-byte reads/writes.
5. **Completion**: the receiver sends `DONE\n` after it has processed every
   inventory entry; the sender's loop exits on reading `DONE` or on
   disconnect.

`push` and `pull` are symmetric roles over this same protocol: `push`
sends the local inventory and then acts as the file **sender**; `pull`
receives the remote inventory and acts as the file **receiver**. There is
no separate negotiation phase distinct from the inventory diff - the whole
"which objects does the peer need" question is answered by walking every
file under `.synapsefs/objects`, `.synapsefs/refs`, `.synapsefs/ref`, and
`.synapsefs/journal` and diffing sizes, once, up front.

## 4. Resumability - size-based, not content-verified

`handle_receive_file()` writes into `<path>.tmp`, opening in append mode
when the request carries `offset > 0`. `receiver_sync_loop()` picks that
offset from the local `.tmp` file's current size (0 if none, or if the
existing `.tmp` is already larger than the remote's reported size, in which
case it restarts from 0). A completed transfer deletes any prior file at
the destination path and renames the `.tmp` file over it.

**A hash is now checked, but only for object files.** After a transfer into
`<path>.tmp` completes (i.e. the accumulated byte count reaches the
inventory's reported size), `handle_receive_file()` checks whether the
destination path matches the loose-object fan-out shape
(`.synapsefs/objects/<2-hex>/<62-hex>`) via `object_oid_from_path()`, which
reconstructs the expected `core::Oid` directly from the two path components
(`"b3:" + dirname + filename`) - no filesystem walk, no trust in the
sender's claims. If it matches, `verify_received_object_payload()` reads the
whole temp file back, decodes it with `format::ObjectHeader::decode()`,
bounds-checks `payload_offset() + payload_len` against the actual file size,
and recomputes `core::compute_oid(hdr->kind, payload)`, comparing it against
the oid derived from the path. On any mismatch - decode failure,
out-of-bounds length, or hash mismatch - the `.tmp` file is deleted, a
message is printed (`"Integrity check FAILED for ... Discarding; this file
was NOT synced."`), and `handle_receive_file()` returns `false` instead of
renaming the file into place. `receiver_sync_loop()` propagates that
failure up through `pull()`'s return value (§6), and `serve()`'s `PUSH`
branch logs a warning telling the operator to run `sfs verify --full` if any
file in an incoming push fails this check.

The whole-payload BLAKE3 recompute is what proves
integrity here, not a chunk-by-chunk walk. See
[`threat_model.md`](../threat_model.md) for what remains unverified.

## 5. Server

`serve(int port)` is a single-threaded, single-connection-at-a-time
`socket()`/`bind()`/`listen(fd, 5)`/`accept()` loop on `INADDR_ANY:port`.
It blocks on one client until that client disconnects before accepting the
next; there is no concurrency. It dispatches on the first line read
(`PUSH`-> act as receiver, `PULL` -> act as sender); any other first line is
silently ignored and the connection closed.

The documented default port is now actually applied. `RepoConfig.listen`
defaults to `"127.0.0.1:9418"`; `apps/sfs/cmd/serve.cpp`'s backing `static
int port` used to be declared with no initializer and the CLI11 option had
no `->default_val(...)` call, so running `sfs serve` with no `-p/--port`
zero-initialized `port` and bound an OS-assigned ephemeral port instead -
matching neither the `9418` `RepoConfig` documented nor the "By Default
8002" the old `--help` text itself claimed (a second, independent
inconsistency). Both are fixed: `port` is now declared `= 9418`, the CLI11
option carries `->default_val(9418)`, and the stale "By Default 8002" text
is gone. Running `sfs serve` with no `-p/--port` now binds `9418`, matching
`RepoConfig::listen`. `-p/--port` still overrides it explicitly when
needed. Note `net::serve()` itself still takes only a raw `int port` and
never reads `RepoConfig` - the fix is at the CLI default, not a new
config-read path - so passing an explicit port that disagrees with
`RepoConfig.listen` is still possible and still silently accepted.

## 6. CLI-level exit-code caveat

`run_push()`/`run_pull()` (`apps/sfs/cmd/push.cpp`/`pull.cpp`) now check the
returned `bool` and return `ExitCode::Network` with a stderr message on
failure. For `pull` specifically, "failure" now includes the §4 integrity
check rejecting a received object, not just a dropped connection. A failed
push or pull is now visible at the shell exit-code level for
connection/transfer failures and (for pull) received-object integrity
failures.
