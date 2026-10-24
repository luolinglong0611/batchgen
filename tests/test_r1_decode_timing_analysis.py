"""Archived timing attribution must not mix ranks, phases or timezones."""
import datetime as dt
import importlib.util
from pathlib import Path
import unittest

FILE = Path(__file__).resolve().parents[1]/'scripts/r1_reproduction/analyze_decode_timing.py'
SPEC = importlib.util.spec_from_file_location('r1_decode_attribution',FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DecodeTimingAttribution(unittest.TestCase):
    def test_clips_windows_and_excludes_other_ranks_and_warmup(self):
        log = '\n'.join([
            '2026-10-24 16:00:00,000 - [BatchGenWorker-0] - [DECODE] Prepared batch: 1 sequences',
            '2026-10-24 16:00:05,000 - [BatchGenWorker-0] - [DECODE] step=1 step_ms=20.0 (fwd_launch 19.0) gpu_fwd_ms=18.0 (n=4)',
            '2026-10-24 16:00:06,000 - [BatchGenWorker-1] - stage=deep_free_enter',
            '2026-10-24 16:00:07,000 - [BatchGenWorker-0] - stage=deep_free_enter',
            '2026-10-24 16:00:08,000 - [BatchGenWorker-0] - [PREFILL] Selected 3 sequences (2 recompute from eviction)',
            '2026-10-24 16:00:09,000 - [BatchGenWorker-0] - deep_free_model_memory sync=0.1s del=0.2s release_streamed=0.3s empty_cache=0.4s gc=0.5s',
            '2026-10-24 16:00:10,000 - [BatchGenWorker-0] - [DECODE] Prepared batch: 3 sequences',
            '2026-10-24 16:00:13,000 - [BatchGenWorker-0] - [PREFILL] Selected 99 sequences (99 recompute from eviction)',
        ])
        measurement={'started_utc':'2026-10-24T08:00:02+00:00','wall_seconds':10.,'upload_and_create_preparation_seconds':.1}
        result=MODULE.analyze(log,measurement)
        self.assertEqual(result['rank0_wall_windows_seconds'],{'decode_window':7.,'other':3.})
        self.assertEqual(result['logged_decode_steps'],4)
        self.assertEqual(result['recomputed_admissions'],2)
        self.assertAlmostEqual(result['rank0_model_free_wall_seconds'],1.5)
        self.assertEqual(result['weighted_decode_timing_ms']['step_ms'],20.)

    def test_no_timing_events_never_imply_gpu_work(self):
        result=MODULE.analyze('',{'started_utc':'2026-10-24T00:00:00+00:00','wall_seconds':5.})
        self.assertEqual(result['rank0_wall_windows_seconds'],{'decode_window':0.,'other':5.})
        self.assertIsNone(result['weighted_decode_timing_ms']['forward_cuda_event_ms'])


if __name__=='__main__':unittest.main()
