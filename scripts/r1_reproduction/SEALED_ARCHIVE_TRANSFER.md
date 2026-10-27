# Transfer one sealed node archive

`transfer_r1_sealed_archive.py` is a standalone standard-library CPU helper.
Transfer the already-sealed node bundle containing its manifest, raw JSONL and
referenced native FP32 files as **one exact archive file**. This helper neither
creates nor extracts archives and does not modify a collector run, suite,
deployment, timestamps, HTTP records or evidence thresholds. The existing
cross-node finalizer still checks the manifests and complete native evidence.

All commands default to plan-only. `--execute` is required for token creation,
listening or receiving. The helper is small enough for the parent's bounded
script upload; archive bytes travel directly between the two explicit node IPs.
No remote command or listener was used to develop or test it.

The parent creates a token and supplies its exact bytes to a private file on
each node through its existing private configuration channel. Do not print the
token or put it in a URL or command-line argument. Only the token-file paths
appear in these commands:

```text
python transfer_r1_sealed_archive.py token --execute
  --output /ABSOLUTE/FRESH/PRIVATE/token
```

The token has 256 random bits and its file is mode 0600. The source creates a
new `token.used` marker before the first authenticated transfer. Failed and
interrupted authorized attempts also consume the token; retries need a fresh
token and marker path. Unauthorized requests do not consume it.

Source example, sending node1's already-sealed archive to node0:

```text
python transfer_r1_sealed_archive.py serve --execute
  --archive /ABSOLUTE/SEALED/node1.tar
  --sha256 PINNED_ARCHIVE_SHA256 --bytes PINNED_ARCHIVE_BYTES
  --source-ip 10.66.9.166 --receiver-ip 10.66.9.163 --port 30201
  --token-file /ABSOLUTE/FRESH/PRIVATE/token --timeout 600
```

Receiver:

```text
python transfer_r1_sealed_archive.py receive --execute
  --output /ABSOLUTE/EXISTING/DIRECTORY/FRESH-node1.tar
  --sha256 SAME_PINNED_ARCHIVE_SHA256 --bytes SAME_PINNED_ARCHIVE_BYTES
  --source-ip 10.66.9.166 --receiver-ip 10.66.9.163 --port 30201
  --token-file /ABSOLUTE/FRESH/PRIVATE/token --timeout 600
```

The source binds only its literal IP, accepts only the literal receiver IP and
serves only `GET /archive` with the exact Bearer token. There is no directory
listing, query-selected path, range selection, command execution or extraction.
The receiver explicitly binds its outbound socket to its receiver IP and
connects directly to the literal source IP; it uses neither proxies nor DNS.
HTTP is the requested internal-network transport; this helper adds no TLS.

Before listening, the source opens the one ordinary file without following a
leaf symlink and verifies its pinned size and complete SHA. It retains that
file descriptor, verifies pathname and descriptor stat identities before and
after streaming, and hashes the stream again. Source receipts include both
stat identities, never the token. A source success only means bytes were sent;
the receiver's independently verified receipt establishes completed receipt.

The receiver requires the exact declared length/SHA headers and verifies the
entire streamed file. It writes a fresh mode-0600, exclusive temporary file in
the destination directory. After length/SHA and fsync, it atomically publishes
with an exclusive hard link and removes the temporary name. This intentionally
avoids an overwrite-capable rename: an existing or concurrently created final
path always causes failure without changing that file. Ordinary transfer
failures remove the temporary file; an external SIGKILL can leave an unverified
`.part-*` file, which is never accepted as the final archive.

Wrap source and receiver in the existing parent-owned process monitor for a
hard wall limit and lifecycle receipts. The source stops after one authorized
attempt or its deadline; per-connection header reads are bounded. The receiver
has network timeouts and checks its deadline while streaming. A fresh receive
path and fresh token are required for a new attempt. Do not interpret transfer
success as correctness or performance qualification.

CPU checks (mocked HTTP connections; no port binding):

```text
python -m unittest discover -s tests -p test_r1_sealed_archive_transfer.py -v
```
