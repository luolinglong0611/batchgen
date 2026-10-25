import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('prefill_analysis', Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction/analyze_r1_prefill_weights.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    stamp = '2026-10-25 20:00:00,000'
    unix = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    rows = []
    for stage, layers in [('attention', range(61)), ('routed_moe', range(3, 61)), ('dense_mlp', range(3)), ('prefill_weight_wait', (3, 30, 60)), ('prefill_expert_compute', (3, 30, 60)), ('prefill_weight_release', (3, 30, 60))]:
        rows += [{'phase': 'prefill', 'rank': 0, 'call': 1, 'stage': stage, 'layer': layer, 'unix': unix, 'cpu_ms': 100, 'stream_ms': 50} for layer in layers]
    counters = {'h2d': {'enabled': True, 'by_type': {'routed_expert': {'bytes': 1234, 'copy_s': 1.5}}}, 'consumer_wait': {'enabled': True, 'by_type': {'routed_expert': {'wait_s': .5}}}}
    return rows, f'{stamp} - [R1_WEIGHT_STREAM] rank=0 ' + json.dumps(counters), unix


class PrefillWeightAnalysisTest(unittest.TestCase):
    def test_nested_spans_and_actual_copy_counters_stay_separate(self):
        rows, log, unix = fixture()
        result = module.analyze(rows, log, unix - 1, unix + 1, ranks=1)
        sample = result['per_rank']['0'][0]
        self.assertAlmostEqual(sample['forward_stream_seconds']['routed_moe'], 2.9)
        self.assertEqual(sample['selected_layers']['3']['prefill_weight_wait']['host_seconds'], .1)
        self.assertEqual(result['native_weight_stream']['0']['counters']['h2d']['by_type']['routed_expert']['bytes'], 1234)

    def test_partial_layers_or_missing_weight_spans_rejected(self):
        rows, log, unix = fixture()
        for stage in ('attention', 'prefill_expert_compute'):
            broken = [r for r in rows if not (r['stage'] == stage and r['layer'] == 3)]
            with self.assertRaises(ValueError):
                module.analyze(broken, log, unix - 1, unix + 1, ranks=1)

    def test_nonfinite_or_negative_event_duration_rejected(self):
        rows, log, unix = fixture()
        for value in (float('nan'), float('inf'), -1):
            broken = [dict(r) for r in rows]
            broken[0]['stream_ms'] = value
            with self.assertRaises(ValueError):
                module.analyze(broken, log, unix - 1, unix + 1, ranks=1)

    def test_duplicate_or_out_of_window_native_counters_rejected(self):
        rows, log, unix = fixture()
        with self.assertRaises(ValueError):
            module.analyze(rows, log + '\n' + log, unix - 1, unix + 1, ranks=1)
        with self.assertRaises(ValueError):
            module.analyze(rows, '', unix - 1, unix + 1, ranks=1)
        with self.assertRaises(ValueError):
            module.analyze(rows, log, unix + 1, unix + 2, ranks=1)


if __name__ == '__main__':
    unittest.main()
