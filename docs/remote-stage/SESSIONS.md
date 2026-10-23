# Session files across a remote stage

Status: built and unit-tested on the CPU (no GPU, no model). Not yet run on two GPUs: nothing below is measured.
Opt-in as on one card: the server needs `--slot-save-path`, the main process `--prompt-cache N`.

## The problem

A session file (`POST /slots/0?action=save|restore`, [DETAILS.md](../DETAILS.md)) holds the conversation the engine
holds: the running state, the deepest checkpoint and every attention layer's K/V. With `--remote-stage` the main
process holds layers 0..K-1 and the worker holds K and after, so a file of the main process alone would miss half the
model. Until now a remote main refused session files ("session files do not support --remote-stage").

## What it does

One file on the main PC holds the whole conversation. The worker's layers cross the link during the save and the
restore, a block at a time; the worker never touches a disk and never reads a path from the wire. A restore needs no
file on the worker, so it works with a worker that was started again, or another PC with the same setup.

## The file (format v2)

Format v1, the one-process file, is unchanged byte for byte (the golden test in `conversation_file_test` still holds).
A remote main writes format v2:

| Part | Bytes | What |
|---|---|---|
| Header | 64 | as v1, with version 2 and the worker part's size in the first reserved field (offset 40) |
| Worker part | as the header says | the worker's own bytes (below), passed through, never parsed by the main process |
| Main payload | the rest | v1's payload of layers 0..K-1, with the draft layer |
| Trailer | 16 | v1's: the payload hash (over the worker part and the main payload) and the end marker |

The worker part: magic `STRSPRT\x01`, version u32 (1), header size u32 (32), the worker's model and config
fingerprints (u64 each), then v1's payload of the worker's layers (geometry, layer range, cvec flag, the live state,
at most one checkpoint, its K/V layers; no draft layer). It has no hash of its own: the file's hash covers it, and the
main process checks that hash before the worker puts its part in place.

## Identity: which file restores where

| Check | Where | Covers |
|---|---|---|
| Main model fingerprint | main, file header | the main PC's model files, as on one card |
| Main config fingerprint | main, file header | the main's settings as on one card, plus the split point K (`remote_stage_begin`) |
| Worker model and config fingerprints | worker, part header | the worker PC's own files and settings (its `--kv`, `--max-context`, switches) |
| Geometry and layer range | worker, part payload | the worker's layers are K..last |
| Format version | main | v1 on a split, or v2 on one process: refused both ways |

No proof that two layouts hold the same bytes is attempted, so the refusal of v1 on a split and of v2 on one process
is unconditional. The hello already makes both PCs run the same model pack (its fingerprint) and K/V type.

## Protocol

No protocol bump: a worker offers session files with a new capability bit in its hello's `row_formats`
(`kStageCapSession`, bit 30), and the main process sends the new messages only to a worker that has it. Everything
else stays protocol 7, so a main and a worker of different builds still work together, without session files.

| Message | Fields | Payload | Reply |
|---|---|---|---|
| `SessionSave` (16) | a = tokens, b = checkpoint id (0: none), c = its tokens | `int32 ids[a]` | `SessionOk` (a = the part's bytes); the worker holds the part |
| `SessionPull` (17) | | | `SessionData` blocks, then `SessionOk` (a = bytes); or an `Error` in place of any of them |
| `SessionLoad` (18) | a = the part's bytes | | `SessionOk` (it fits); then the main sends `SessionData` blocks; after the last, `SessionOk` (a = tokens, b = checkpoint tokens) |
| `SessionEnd` (19) | a = 1 apply (b = the checkpoint's new id) / 0 drop | | `SessionOk` |
| `SessionData` (20) | c = 1 and no payload: the main gives a load up | 1..16 MiB | |
| `SessionOk` (21) | | | |

A worker holds a part between these messages and lets it go before any other message and at a disconnect. A worker
error starts with its kind (`invalid: `, `memory: `, `io: `), which the main process answers as its own. A Commit
that failed before a `SessionLoad` stays pending (the load may be given up, and the session goes on) until
`SessionEnd` puts the part in place; then it is dropped, as a `CkptRestore` drops it.

## A save

1. The main process checks the link (a worker that went away while idle took its part with it: refused, `link`), the
   worker's capability bit, and that the link was not made again since the last request.
2. Its RAM preflight (the one-card estimate plus one 16 MiB link block) and its own capture, as on one card.
3. `SessionSave` with the live token ids and the deepest checkpoint's worker id. The worker checks that its session
   is at that many tokens and that it holds that checkpoint at that length, runs its own RAM preflight (above
   `--conversation-cache-min-free-mib`), captures its running state and holds the part (K/V read later, in place).
4. The file write: the free-space preflight (`--session-min-free-mib`) covers the whole file, worker part included.
   Then `SessionPull` streams the part into the file as it comes, then the main payload, the hash, the rename.

## A restore

1. A link the worker's end closed while idle is made again first (the restore replaces the session anyway). Then the
   capability bit.
2. The file header: version, both fingerprints, sizes. Then `SessionLoad` with the part's size: the worker checks it
   against the largest part its session can restore (its layers, its context, one checkpoint) and its RAM before
   any byte is sent. The part streams to the worker as the file is read; the worker parses it with every bound a
   file read has (identity before the payload, every count before its allocation, every byte consumed) and validates
   it against its session, without a device write, and holds it.
3. The main payload, then the file's hash over both parts. Both parts must hold the same tokens and checkpoint.
4. The main validates its own image. Then `SessionEnd` apply: the worker puts its part in place, empties its
   checkpoint store and keeps the file's checkpoint under the main's new id. Then the main's own transfer (a failure
   there ends the engine, `FATAL`, as on one card).

## Failures

| What fails | Answer | After it |
|---|---|---|
| The file, the identity, a size or a bound (either side) | `invalid` (400) | both sessions as they were; a held part is dropped |
| RAM on either side | `memory` (503) | as they were |
| The disk | `storage` (507) | as they were |
| The link, during a save or a restore | `link` (503) | the engine stays; the next request connects again and reads its prompt from token 0 |
| The worker's own step (a device copy, a position that does not match) | `io` (500) | as they were, or, at the worker's apply, the session dropped on both sides |
| The main's transfer after the device writes began | `FATAL` (the server answers 500 and starts the engine again) | as on one card |

A side that fails part-way keeps the link in step: the worker answers an `Error` in place of its next block, the main
gives a load up with an empty `SessionData` (c = 1), and the worker drains the rest of a part it refused. A file that
fails under a save still reads the rest of the worker's part from the link.

## Untrusted peer

Each side bounds a size before any product or allocation with it: a `SessionSave`'s token count (1..2^22) before its
ids, a block's size (1..16 MiB, and never past what is left), a part's size against the worker's limits before any
of it is read. At most two blocks wait in the worker's reader (a part is gigabytes). A malformed or misplaced
`SessionData` ends the connection unread: nothing after it is in step. A malformed reply on the main ends the link,
not the engine (the command answers `link`).

## Relay workers

A relay worker (`--stage-next`) does not forward session files. Its hello does not offer them, so the main process
refuses them itself ("does not take session files (a relay worker forwards none ...)"); the relay logs a line at its
start, and answers an error if asked anyway.

## Unchanged

- One card and an in-process `--layer-split`: the same file format, the same bytes, the same messages.
- A remote stage without SAVE or RESTORE: no new message crosses the link.
- `--layer-split` (in-process), `--peer-device` and `--batch` still refuse session files.

## As built

- `include/strata/core/conversation_file.hpp`, `src/core/conversation_file.cpp`: format v2 (`SessionWriteOptions.stage`,
  `SessionReadLimits.stage`), the worker part (`session_part_bytes / _write / _max_bytes / _read`), `SessionError::link`.
- `include/strata/net/stage_link.hpp`, `src/net/stage_link.cpp`: the messages, `StageClient::session_save / _pull /
  _load / _end`, the worker's `StageHandlers` and their serving.
- `src/program/generate.cpp`: the main's SAVE / RESTORE with a remote stage, the worker's handlers, the capability
  bit, the split point in the config fingerprint (on a remote main only).
- `serve/server.py`: `link` answers 503.

Tests (no GPU): `conversation_file_test` (v2 round trip with a part over 16 MiB in uneven pieces, the header fields,
the refusals both ways, a part over its limit refused before it is read, a flipped byte in the part, stream failures
short, long and the disk's, the part's own reader: identity, truncation, extra bytes, size limit, RAM preflight,
oversized count), `stage_link_test` section 17 (every message, both sides failing part-way, malformed messages),
`serve/test_slots.py` (`link` is 503).

## Not done

- Not run on GPUs yet: no save/restore time, file size or answer comparison is measured here.
- A file restores only on the same split; moving a conversation between a split and one card is not supported.
- Relay chains (three PCs or more) are refused.
- Two main processes pointed at one worker take turns after `STRATA_REMOTE_TAKEOVER_S` of silence. A restore is
  silent on the link while the main process reads its own part of the file and checks it; if another main process
  takes the worker over then, the restore fails as `link`. Not seen, reasoned from the code.
- The main process's `generate.cpp` glue (the order of the steps, the cross-checks, the new checkpoint id) has no
  unit test; the GPU runs are its test.
