# R1 native completion token diagnostics

Set `BATCHGEN_R1_NATIVE_TOKEN_TRACE=1` in the BatchGen worker environment for
`model_name=deepseek-ai/DeepSeek-R1`. The default is off; other model names and
other flag values stay off. This is an opt-in diagnostic for a short correctness
probe, not a benchmark setting. It has not been deployed by this change.

`_gather_completed_tokens` logs one JSON record prefixed `[R1_NATIVE_TOKENS]`
for each locally bound sequence on its assigned rank, before completion releases
the trajectory. It reuses the CPU generated-token tensor already used for text
output and reads the original prompt prefix from the still-active CPU trajectory.
The disabled path performs no additional trajectory read or tensor processing.
Sampling, output text, API responses and distributed gather payloads are unchanged.

The `r1_native_tokens_v1` record identifies `rank`, `assigned_rank`, `uuid`,
`batch_id` and `global_idx`. `generated_ids` contains native sampled IDs before
EOS/special-token trimming, in turn-ledger order, including earlier generated
spans after re-entry and excluding subsequent user prompts. At most 8192 IDs
are logged. `generated_ids_count`, `generated_ids_logged_count`,
`generated_ids_omitted_count` and `generated_ids_truncated` make the exact scope
explicit. `decoded_length` is the sequence's actual completion metadata;
`decoded_length_matches_ids_count` exposes a discrepancy instead of masking it.

`original_prompt_ids_sha256` hashes the full original prompt IDs, encoded as
UTF-8 JSON with compact separators (for example `[11,22]`), with no whitespace,
newline or dtype/endianness metadata. `prompt_ids_hash_encoding` names this
encoding. `original_prompt_length` and `original_prompt_ids_count` must agree.
Later internal prompt reconstruction does not change this hash. Raw prompt IDs
are not logged.

`finish_reason` uses the completion handler's precedence: repetition, decode
budget, context limit, EOS flag, then length fallback. The classifier has no
lifecycle logging side effects. Unexpected devices, invalid tensors, prompt
count mismatches or diagnostic failures produce `[R1_NATIVE_TOKENS_ERROR]`
with rank, UUID and exception class, while the existing text/collective path
continues. An absent or failed trace is missing diagnostic evidence, not a
successful native-token comparison. No device transfer or CUDA query is added.
Compare records within the specific service-run log: UUIDs can be reused across
runs, and this record does not add a run ID. Exceptional completions that bypass
`_gather_completed_tokens` do not produce this trace.

CPU verification:

```sh
python -m pytest -q tests/test_r1_native_token_trace.py
```

These tests execute the real helper with CPU tensors and the extracted worker
completion method with the real CPU QueryBook. They cover default-off behavior,
owner/lifetime boundaries, re-entry spans, EOS retention, the 8192-ID cap,
finish-reason parity and diagnostic failure containment. They do not qualify
multi-GPU correctness or the production deployment.
