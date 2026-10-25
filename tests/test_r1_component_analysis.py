import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('component_analysis',Path(__file__).parents[1]/'scripts/r1_reproduction/analyze_r1_components.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def complete_rows(rank=0):
    rows=[]
    for name,count in module.EXPECTED.items():
        for index in range(count):
            layer=index//2+3 if name=='quantize'else index+(3 if count==58 else 0)
            rows.append({'schema':'r1-component-stream-interval-v1','rank':rank,'phase':'decode',
                         'call':32,'stage':name,'layer':layer,'unix':100,'cpu_ms':.01,'stream_ms':.1})
    return rows

def test_reject_partial_and_keep_ranks_separate():
    rows=complete_rows()+complete_rows(8)
    result=module.analyze(rows,99,101)
    assert set(result['per_rank'])=={'0','8'}
    assert all(x['complete_steps']==1 for x in result['per_rank'].values())
    result=module.analyze(rows[:-1],99,101)
    assert set(result['per_rank'])=={'0'}
    assert result['rejected_partial_decode_samples'][0]['rank']==8

def test_no_warmup_or_parent_child_double_count():
    rows=complete_rows()
    rows+=[{**r,'unix':50}for r in complete_rows()]
    result=module.analyze(rows,99,101)
    assert result['per_rank']['0']['complete_steps']==1
    assert result['per_rank']['0']['median_observed_forward_stream_ms']==pytest.approx(12.2)

def test_duplicate_layers_and_nonfinite_rejected():
    rows=complete_rows();rows[-1]['layer']=rows[-2]['layer']
    assert not module.analyze(rows,99,101)['per_rank']
    rows[-1]['stream_ms']=float('nan')
    with pytest.raises(ValueError,match='Non-finite'):module.analyze(rows,99,101)


def test_native_prefill_max_uses_coverage_and_excludes_warmup():
    import datetime as dt
    from zoneinfo import ZoneInfo
    unix=dt.datetime(2026,10,25,12,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    rows=[]
    for rank,seconds in [(0,10),(8,15)]:
        import json
        payload={'component':'prefill_full_attention','rank':rank,'sequences':32,'prompt_tokens':65536,
                 'prefill_forward_host_s':seconds,'prefill_forward_gpu_complete_s':seconds+.1,'prefill_kv_ready_s':seconds+.2}
        rows.append(f'2026-10-25 12:00:00,000 - [BatchGenWorker-{rank}] - INFO - [METRICS] '+json.dumps(payload))
    rows.append(rows[0].replace('12:00:00','11:00:00'))
    result=module.analyze_native_phases('\n'.join(rows),unix-1,unix+1)
    assert result['prefill_participating_ranks']==[0,8]
    assert result['prefill_total_sequences']==64
    assert result['max_rank_prefill_seconds']['prefill_kv_ready_s']==15.2


def test_zero_token_expert_rank_is_complete_with_zero_gemm_work():
    optional={'compact','active_experts_readback','gate_up_gemm','down_gemm','quantize'}
    rows=[r for r in complete_rows()if r['stage']not in optional]
    result=module.analyze(rows,99,101)
    assert result['per_rank']['0']['complete_steps']==1
    assert result['per_rank']['0']['stages']['down_gemm']['stream_ms']==0


def graph_rows():
    rows=[r for r in complete_rows() if r['stage'] in module.GRAPH_EXPECTED]
    rows += [{**r, 'stage':'local_moe_graph'} for r in rows if r['stage']=='routed_moe']
    return rows


def test_graph_replay_keeps_inner_stages_unobserved():
    result=module.analyze(graph_rows(),99,101)
    rank=result['per_rank']['0']
    assert rank['execution_paths']==['local_graph_replay']
    assert rank['stages']['local_moe_graph']['stream_ms']==pytest.approx(5.8)
    assert 'gate_up_gemm' not in rank['stages']
    assert 'token_count_readback' not in rank['stages']


def test_partial_duplicate_and_fallback_graph_samples_are_rejected():
    rows=graph_rows()
    assert not module.analyze(rows[:-1],99,101)['per_rank']
    rows[-1]['layer']=rows[-2]['layer']
    assert not module.analyze(rows,99,101)['per_rank']
    fallback=graph_rows()+[r for r in complete_rows() if r['stage']=='gate' and r['layer']==3]
    assert not module.analyze(fallback,99,101)['per_rank']
    no_graph=[r for r in graph_rows() if r['stage']!='local_moe_graph']
    assert not module.analyze(no_graph,99,101)['per_rank']
