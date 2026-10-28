"""One-GPU native R1 DeepGEMM operator qualification; parent alone may execute.

Default is a stdlib-only plan. No model loading, service, dispatch or benchmark
queue is started. Numerical references independently dequantize the *converted*
FP8/UE8M0 operands, so this does not qualify original-weight/SG model precision.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    assert args.stride >= 64 and args.stride % 4 == 0
    assert 1 <= args.timing_repeats <= 100
    source = args.repo / "batchgen/models/deepseek/deepseekv3/deepgemm_moe.py"
    pin = hashlib.sha256(source.read_bytes()).hexdigest()
    if not args.execute:
        print(json.dumps({"status": "plan_NOT_GPU_qualified", "source": str(source), "sha256": pin,
            "shape": {"experts": 32, "hidden": 7168, "intermediate": 2048, "stride": args.stride},
            "reference_experts": [0, 1, 7, 31], "counts": "64 each; 0/1/32/64 mixed; all zero",
            "graph": "changed FP8 input and live counts, exact eager/replay comparison",
            "tolerances": {"S1_NRMSE": 0.025, "S3_NRMSE": 0.03},
            "scope": "one GPU math only; no DeepEP transport/full model/SG accuracy qualification"}, indent=2))
        return
    import torch
    assert torch.cuda.device_count() == 1, "Parent must reserve one exclusive visible GPU"
    assert torch.cuda.get_device_capability(0) == (10, 3)
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(310028)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = importlib.util.spec_from_file_location("r1_native_deepgemm_probe", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    e, h, n, m = 32, 7168, 2048, args.stride
    rt = module._load_runtime("cuda:0")

    def host_sources():
        for _ in range(e):
            item = {}
            for name, shape in (("gate", (n, h)), ("up", (n, h)), ("down", (h, n))):
                item[name] = torch.randint(-16, 17, shape, dtype=torch.int8).to(torch.float8_e4m3fn)
                item[name + "_scale"] = torch.rand((shape[0] // 128, shape[1] // 128)) * 0.003 + 0.001
            yield item

    started = time.monotonic()
    math = module.R1DeepGemmExpertMath.from_expert_sources(
        host_sources(), device="cuda:0", source_identity="random_operator_only", runtime=rt)
    p = SimpleNamespace(num_local_experts=e, stride=m, intermediate=n,
        x=torch.empty((1, h), dtype=torch.bfloat16, device="cuda"),
        intermediate_fp8=torch.empty((e, m, n), dtype=torch.float8_e4m3fn, device="cuda"),
        intermediate_scale=torch.empty((e, m, n // 128), dtype=torch.float32, device="cuda"))
    work = module.R1DeepGemmMaskedWorkspace.for_pool(p)
    out = torch.full((e, m, h), -17, dtype=torch.bfloat16, device="cuda")
    eager = torch.empty_like(out)
    counts = torch.full((e,), 64, dtype=torch.int32, device="cuda")
    bf16_input = torch.randn((e, m, h), dtype=torch.bfloat16, device="cuda") * 0.125

    def quantize_expert_input(value):
        # The installed stock SG native quantizer supports 2-D scales only.
        # Its flattened MN-major scale has K stride E*M, so a view alone is
        # NOT a valid per-expert DeepEP/DeepGEMM scale layout. Copy logical
        # [E,M,K/512] entries into independent MN-major expert slices.
        quantized, flat_scale = rt.quantize(
            value.reshape(e * m, h), 128, column_major_scales=True,
            scale_tma_aligned=True, scale_ue8m0=True)
        assert flat_scale.dtype == torch.int32
        expert_scale = torch.empty((e, h // 512, m), dtype=torch.int32,
                                   device=value.device).transpose(1, 2)
        expert_scale.copy_(flat_scale.view(e, m, h // 512))
        assert expert_scale.stride() == (m * (h // 512), 1, m)
        return quantized.view(e, m, h), expert_scale

    x, sf = quantize_expert_input(bf16_input)
    assert x.dtype == torch.float8_e4m3fn and sf.dtype == torch.int32
    calls = {"native_masked": 0, "old_S1_S3_imported": False}
    raw_masked = rt.dg.fp8_m_grouped_gemm_nt_masked

    def counted(*args, **kwargs):
        calls["native_masked"] += 1
        return raw_masked(*args, **kwargs)

    # Capture executes Python once, replays do not. Changed GPU outputs prove
    # actual replay execution; this counter is never presented as replay count.
    rt.dg.fp8_m_grouped_gemm_nt_masked = counted
    expected_m = 64

    def forward(target):
        return math.masked(x, sf, counts, expected_m, workspace=work, out=target)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward(out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    def unpack(scale, k):
        if scale.dtype == torch.int32:
            b = scale.contiguous().view(torch.uint8).reshape(*scale.shape[:-1], -1)[..., :k // 128]
            return torch.pow(2.0, b.int() - 127)
        return scale

    def dequant(values, scale):
        return values.float() * unpack(scale, values.shape[-1]).repeat_interleave(128, -1)

    def error(actual, reference):
        difference = actual.float() - reference.float()
        denominator = reference.float().square().mean().sqrt().clamp_min(1e-6)
        return {"NRMSE": float(difference.square().mean().sqrt() / denominator),
                "max_abs": float(difference.abs().max())}

    metrics = []

    def references(label):
        current_counts = counts.cpu().tolist()
        for expert in (0, 1, 7, 31):
            live = current_counts[expert]
            if live == 0:
                continue
            left = dequant(x[expert, :live], sf[expert, :live])
            w13 = dequant(math.bank.w13[expert], math.bank.w13_scale[expert])
            reference_gateup = (left @ w13.T).to(torch.bfloat16)
            s1 = error(work.gateup[expert, :live], reference_gateup)
            del left, w13, reference_gateup
            # Independent second GEMM reference, using the stock fused
            # activation's real FP8 output and unpacked power-of-two scales.
            down = dequant(work.down_fp8[expert, :live], work.down_scale[expert, :live])
            w2 = dequant(math.bank.w2[expert], math.bank.w2_scale[expert])
            reference_down = (down @ w2.T).to(torch.bfloat16)
            s3 = error(out[expert, :live], reference_down)
            del down, w2, reference_down
            metrics.append({"case": label, "expert": expert, "valid_rows": live, "S1": s1, "S3": s3})
            assert s1["NRMSE"] <= 0.025 and s3["NRMSE"] <= 0.03, metrics[-1]

    references("eager_all64")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(out)
    cases = []
    old_input_signature = float(x[0, :64].float().sum())
    for step, pattern in enumerate(([0, 1, 32, 64] * 8, [64] * 32, [0] * 32)):
        counts.copy_(torch.tensor(pattern, dtype=torch.int32, device="cuda"))
        new_input = torch.randn_like(bf16_input) * (0.2 + 0.05 * step)
        new_x, new_sf = quantize_expert_input(new_input)
        x.copy_(new_x)
        sf.copy_(new_sf)
        del new_input, new_x, new_sf
        eager.fill_(-17)
        forward(eager)
        out.fill_(-17)
        graph.replay()
        torch.cuda.synchronize()
        for expert, live in enumerate(pattern):
            if live:
                assert torch.equal(eager[expert, :live], out[expert, :live]), {
                    "step": step, "expert": expert,
                    "valid_rows_eager_graph_max_abs": float((eager[expert, :live] - out[expert, :live]).abs().max())}
            else:
                assert torch.all(out[expert] == -17), {
                    "step": step, "expert": expert, "zero_expert_output_changed": True}
        # DeepGEMM stores complete M tiles. Rows after a nonzero live count
        # are intentionally undefined and DeepEP must not combine them.
        cases.append({"step": step, "counts": pattern, "valid_rows_eager_graph_bitwise": True,
                      "zero_expert_output_untouched": True,
                      "partial_M_tile_padding": "undefined_NOT_compared_NOT_valid_output",
                      "input_signature": float(x[0, :64].float().sum())})
        references("graph_step" + str(step))
    assert cases[0]["input_signature"] != old_input_signature
    # Timing is isolated math only, on one GPU, after native shape compilation.
    # This must never be extrapolated to model throughput or formal BCT.
    counts.fill_(64)
    events = []
    for _ in range(args.timing_repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    milliseconds = [a.elapsed_time(b) for a, b in events]
    calls["old_S1_S3_imported"] = any(name.endswith("grouped_gemm_kernel") for name in sys.modules)
    assert not calls["old_S1_S3_imported"]
    rt.dg.fp8_m_grouped_gemm_nt_masked = raw_masked
    result = {"status": "passed_operator_only_NOT_model_or_transport_qualified",
        "source_sha256": pin, "shape": {"experts": e, "hidden": h, "intermediate": n, "stride": m},
        "math": "native_DeepGEMM_FP8_UE8M0_two_masked_GEMMs_plus_stock_SG_fused_activation",
        "bank_build": math.build_stats, "references": metrics, "graph_cases": cases,
        "python_capture_and_eager_calls_NOT_replay_count": calls,
        "graph_math_ms": {"samples": milliseconds, "mean": sum(milliseconds) / len(milliseconds)},
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "wall_seconds": time.monotonic() - started,
        "precision_scope": "converted UE8M0 operands; not original FP32-scale or SGLang whole-model equality"}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pin
    payload = json.dumps(result, indent=2)
    if args.result:
        args.result.write_text(payload + "\n")
    print("R1_NATIVE_DEEPGEMM_GPU=" + payload)


if __name__ == "__main__":
    main()
