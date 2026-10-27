# R1 common BF16 attention compatibility candidate

This candidate changes only the unsupported native attention primitives on SM103. It is based on official main `33ce112983f6fae019803b8ae1eb0ef3aae6e8fa`; no old graph, JIT, GC, tile, phase cache or weight scheduling optimization is imported. It is not an exact reproduction of the paper's native attention implementation and has no performance or full-model qualification receipt.

## Model call contract

`batchgen/attention/mla/common_flashinfer_backend.py` exports the same three names as main's native imports. The coordinator selects them only with explicit compatibility configuration. The caller keeps its projection weights, FP8 GEMM, scale, RoPE, absorption, KV append and output projection. `common_sdpa_backend.py` contains shared pure metadata validation and a separate optional Torch FP32 math path; merely importing it does not select that fallback.

Fresh prefill receives already rotated Q/K `[T,H,192]`, V `[T,H,128]`, matching packed Q/K boundaries, and the caller's explicit softmax scale. FlashInfer `BatchPrefillWithRaggedKVCacheWrapper` is forced to `backend="fa2"`, Graph OFF, BF16 input/output, causal full sequences, `pos_encoding_mode="NONE"`. RoPE must not be applied twice. Cached-prefix/asymmetric prefill, biases, softcaps, dropout and probability output are rejected. Query and KV heads must match, avoiding an accidental GQA head mapping.

Decode receives the already absorbed/rotated query `[B,1,H,576]` and contiguous main cache `[P,64,1,576]`: latent512 followed by RoPE64. Cache lengths include the current appended token. The returned latent output is `[B,1,H,512]`; LSE is `[B,H,1]`. Each final query attends to exactly its valid inclusive history. Thus native `causal=True` for one final query maps to FlashInfer `causal=False` over that history. Explicit caller scale is preserved. No extra positional encoding or implicit `1/sqrt(d)` scale is introduced.

## Tail-page behavior

The allocator remains page64. A storage-sharing view reinterprets it as token pages of size1. The FI index list contains only `physical_page*64+valid_offset`, in logical sequence order. Partial last pages and unused table entries are never intentionally addressed. The sliced latent/RoPE views retain the native576 row stride, matching SGLang's FI MLA feature views. The shim never writes, zeros, sanitizes or copies the full cache. NaNs in genuinely valid KV must remain invalid data, rather than being hidden by a tail fix. A GPU fixture is still required to prove that the native kernel obeys these addresses and strides.

## SGLang comparison and required runtime evidence

SGLang v0.5.5.post3's [`flashinfer_mla_backend.py`](https://github.com/sgl-project/sglang/blob/v0.5.5.post3/python/sglang/srt/layers/attention/flashinfer_mla_backend.py) already uses page1 and `causal=False` for final-token MLA, and slices latent/RoPE from a576-wide cache. FlashInfer v0.5.2's [MLA wrapper](https://github.com/flashinfer-ai/flashinfer/blob/v0.5.2/flashinfer/mla.py) supplies the API used here. Matching the package name alone does not establish matching kernels: SG must explicitly select the same `fa2` prefill/decode implementation, Graph OFF, source versions and effective runtime shapes. Export actual installed file hashes and wrapper `_backend` values for both engines. Auto/CUTLASS/native FlashMLA evidence cannot be labeled aligned with this candidate.

## Isolated qualification

`qualify_r1_common_mla.py` defaults to plan-only and requires explicit shim/helper paths and expected SHA256 values. Execution also requires the coordinator-owned `R1_MAIN_MLA_COMPAT_GUARDED=1`, assigned hostname, one visible SM103 GPU, Torch2.9.0+cu130, Triton3.5.0 and FI0.5.2. Torch allocator cap is6GiB; the parent must enforce the advertised external time/host/GPU limits. Qualification loads the exact model shim file and verifies all three entrypoint source locations and helper binding before/after execution.

Normal fixtures cover lengths1,63,64,65,2047,2048,2049,8191,8192,8193 and a packed63/65 prefill. Decode uses shuffled physical pages and poisoned unused pages/tails. Byte hashes cover every cache byte and all inputs. Independent CPU FP64 formulas compute all outputs from exact binary BF16 inputs; CPU reference chunks are32 rows. The frozen envelope is at most1 BF16 representable step from once-rounded FP64 reference, with NRMS/max-absolute error reported. This allowance covers isolated FP32 softmax/accumulation rounding and does not define the composed FP8/model error budget. Unknown failures remain failures; do not widen the envelope after observing them. A separate valid-KV NaN negative fixture must propagate nonfinite output and retain cache bytes.

Only the exact status `passed_isolated_sm103_mla_compat_contract_NOT_model_or_performance_qualified` records this operator fixture pass. It cannot satisfy full-model memory, numerical, semantic or formal performance gates. Model KV lifecycle, mixed request batches, logits/token semantics and BG/SG effective parameter alignment remain independent required checks.
