import importlib.util
import json
from pathlib import Path
import pytest

s=importlib.util.spec_from_file_location('gc_analysis',Path(__file__).resolve().parents[1]/'scripts/r1_reproduction/analyze_r1_decode_gc.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def fixture():
    rows=[]
    for rank in range(2):
        base={'schema':'r1-decode-gc-scope-v1','rank':rank}
        rows += [{**base,'event':'collection','unix':20.5,'generation':2,'wall_ms':5,'cpu_ms':4,'frozen':False},
                 {**base,'event':'activated','unix':21,'forward_call':1,'frozen_objects':100,'automatic_collection_enabled':True},
                 {**base,'event':'collection','unix':25,'generation':2,'wall_ms':.2,'cpu_ms':.1,'frozen':True},
                 {**base,'event':'closed','unix':31,'forward_call':4,'originally_frozen_objects':100,'remaining_frozen_objects':0,'automatic_collection_enabled':True,'generations':[dict(collections=0,wall_ms=0,cpu_ms=0,maximum_wall_ms=0,collected=0)]*2+[dict(collections=2,wall_ms=5.2,cpu_ms=4.1,maximum_wall_ms=5,collected=3)]}]
    return '\n'.join('[R1_DECODE_GC] '+json.dumps(r)for r in rows)


def test_scope_coverage_and_later_teardown_are_separate_from_BCT():
    r=m.analyze(fixture(),20,30,4,expected_ranks=2)
    assert r['per_rank']['0']['closed_after_primary_BCT_seconds']==1
    assert r['across_rank']['while_frozen_gen2_wall_ms']['median']==.2


def test_missing_close_duplicate_scope_and_unrestored_policy_rejected():
    log=fixture()
    for bad in (log.replace('"remaining_frozen_objects": 0','"remaining_frozen_objects": 9'),
                log+'\n'+next(l for l in log.splitlines()if '"event": "activated"'in l),
                '\n'.join(l for l in log.splitlines()if '"event": "closed"'not in l)):
        with pytest.raises(ValueError):m.analyze(bad,20,30,4,expected_ranks=2)


def test_bad_counters_and_incomplete_native_forward_coverage_rejected():
    for bad in (fixture().replace('"wall_ms": 0.2','"wall_ms": -1'),fixture().replace('"forward_call": 4','"forward_call": 3')):
        with pytest.raises(ValueError):m.analyze(bad,20,30,4,expected_ranks=2)
