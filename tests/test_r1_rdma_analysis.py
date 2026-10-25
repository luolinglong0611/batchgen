import copy
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('rdma_analysis',Path(__file__).parents[1]/'scripts/r1_reproduction/analyze_r1_rdma.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def snapshot(unix,data):
    return {'hostname':'host0','observed_unix':unix,'devices':[{
        'device':'mlx5_bond_0','node_guid':'actual-guid','ports':[{
            'port':'1','state':'4: ACTIVE','rate':'200 Gb/sec (2X NDR)',
            'counters':{'port_xmit_data':data,'port_rcv_data':data*2}}]}]}

def test_units_and_full_duplex_capacity_are_separate():
    first,last=snapshot(100,0),snapshot(110,250_000_000)
    result=module.analyze([first,last],{'host0':first},99,111)
    port=result['intervals'][0]['ports'][0]
    assert port['tx_bytes']==1_000_000_000
    assert port['tx_GB_per_second']==.1
    assert port['rx_GB_per_second']==.2
    assert port['reported_port_GB_per_second']==25
    assert port['tx_line_fraction']==.004
    assert port['rx_line_fraction']==.008

def test_no_interpolation_across_warmup_or_measurement_boundary():
    rows=[snapshot(90,0),snapshot(100,10),snapshot(110,20),snapshot(120,30)]
    result=module.analyze(rows,{'host0':rows[0]},99,111)
    assert len(result['intervals'])==1
    assert result['hosts']['host0']['covered_seconds']==10

def test_reset_and_unbound_ports_are_rejected():
    a,b=snapshot(100,10),snapshot(110,9)
    with pytest.raises(ValueError,match='reset'):module.analyze([a,b],{'host0':a},99,111)
    b=snapshot(110,20);b['devices'][0]['node_guid']='different-device'
    with pytest.raises(ValueError,match='identities'):module.analyze([a,b],{'host0':a},99,111)
