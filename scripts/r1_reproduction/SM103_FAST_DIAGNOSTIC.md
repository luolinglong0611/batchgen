# Explicit SM103 fast full-model diagnostic

The R1 launcher defaults to `triton_precision`. To collect full-model diagnostic
outputs with FA4 prefill and FlashInfer MLA decode, select
`--attention-implementation fa4_flashinfer`. The plan then explicitly sets all
three opt-ins:

```
BATCHGEN_R1_SM103_EXPERIMENTAL=1
BATCHGEN_R1_SM103_ATTENTION=fa4_flashinfer
BATCHGEN_R1_SM103_FAST_DIAGNOSTIC=1
```

This selection does not qualify model accuracy or performance. The same-input
attention-only diagnostics retain the original strict gate (`atol=1e-5`,
`rtol=0.016`, failure fraction below `1e-4`). Precision passed the tested rows.
FA4 prefill at lengths 63/64/65 failed with fraction about 0.01897; FlashInfer
decode at 2047/2048/2049 failed at about 0.03200, and at 8191/8192/8193 at about
0.03746. The fast candidates were finite and byte-identical across repeats for
zero and finite padding. With deliberately poisoned NaN padding, FlashInfer
produced nonfinite outputs in both nonaligned decode rows at each length family;
precision remained finite. These failures remain unresolved and preserved.

Preflight still rejects CUDA graphs and prefix-cache reuse. It verifies the
actual FA4 and FlashInfer entry points, pins FA4 4.0.0b36, FlashInfer Python/cubin
0.6.13 and Cutlass DSL 4.8.1, and retains original FP8 weights, FP32 block scales,
CUDA 13 and native-library checks. There is no implicit backend fallback. This
option changes only attention selection; topology, host KV capacity and the
normal server command remain the same.

Use an independently reviewed clean BatchGen environment when launching: the
launcher retains inherited variables outside its explicit environment overrides.
Keep skeleton mmap off for the first full-model probe, record native completion
IDs with `BATCHGEN_R1_NATIVE_TOKEN_TRACE=1`, and inspect the actual run's rank
coverage and padding assumptions. Neither a successful startup nor finite text
is evidence that the numerical or model-quality gates passed. Diagnostic timings
are not the formal 6000-request BatchGen/SGLang comparison.
