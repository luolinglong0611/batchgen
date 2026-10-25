from datetime import datetime
import importlib.util
from pathlib import Path
from zoneinfo import ZoneInfo
import pytest

spec=importlib.util.spec_from_file_location('windows',Path(__file__).resolve().parents[1]/'scripts/r1_reproduction/analyze_r1_decode_windows.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_weighted_native_segments_and_exact_coverage():
    stamp='2026-10-25 20:00:00,000';unix=datetime.strptime(stamp,'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    line=stamp+' - [BatchGenWorker-0] - [DECODE] step_ms=10.0 (setup 0.1, fwd_launch 6.0, kv_launch 0.1, readback 3.5, kv_drain 0.1, bookkeeping 0.2) gpu_fwd_ms=9.5 gpu_fwd_max=15.0 (n=4)'
    r=m.analyze(line,unix-1,unix+1,4);assert r['weighted_step_ms']==10 and r['rows'][0]['logical_steps']==[1,4]
    assert r['weighted_host_split_ms']['readback']==3.5
    with pytest.raises(ValueError):m.analyze(line,unix-1,unix+1,5)
    with pytest.raises(ValueError):m.analyze(line.replace('Worker-0','Worker-1'),unix-1,unix+1,4)
