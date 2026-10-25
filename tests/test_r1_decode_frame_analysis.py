import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('analysis',Path(__file__).resolve().parents[1]/'scripts/r1_reproduction/analyze_r1_decode_frames.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def fixture():
    lines=[]
    for rank in range(2):
        rows=[{'call':c,'unix':10+c/1000,'forward_host_ms':5+rank,'forward_thread_cpu_ms':4,'forward_off_cpu_ms':1+rank,'forward_stream_ms':8,'runqueue_wait_ms':.2+rank}for c in(16,48)]
        lines.append('[R1_DECODE_FRAMES] '+json.dumps({'schema':'r1-decode-frames-v1','rank':rank,'samples':rows}))
    return '\n'.join(lines)


def test_local_observations_and_component_probe_exclusion():
    r=a.analyze(fixture(),10,11,expected_ranks=2)
    assert r['per_rank']['1']['statistics']['all']['forward_host_ms']['mean']==6
    assert r['per_rank']['1']['statistics']['without_component_probes']['forward_host_ms']['count']==1
    assert all(row['max_host_rank']==1 and row['median_stream_ms']==8 for row in r['matched_rank_frames'])


def test_duplicate_missing_rank_and_bad_time_rejected():
    for log,start,end in((fixture()+'\n'+fixture(),10,11),(fixture().splitlines()[0],10,11),(fixture(),11,10)):
        with pytest.raises(ValueError):a.analyze(log,start,end,expected_ranks=2)


def test_invalid_intervals_rejected_instead_of_becoming_zero():
    log=fixture().replace('"forward_host_ms": 5','"forward_host_ms": -5')
    with pytest.raises(ValueError):a.analyze(log,10,11,expected_ranks=2)
