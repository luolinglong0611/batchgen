import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('frames',Path(__file__).resolve().parents[1]/'batchgen/profiling/r1_decode_frames.py')
frames=importlib.util.module_from_spec(spec);spec.loader.exec_module(frames)


class Event:
    def __init__(self,**kwargs):self.complete=False;self.records=[]
    def record(self,stream):self.records.append(stream)
    def query(self):return self.complete
    def elapsed_time(self,other):
        assert self.complete and other.complete
        return 2.5
    def synchronize(self):raise AssertionError('probe must not synchronize')


def test_cpu_runqueue_and_device_intervals_stay_separate():
    events=[]
    def event(**kwargs):
        value=Event(**kwargs);events.append(value);return value
    torch=SimpleNamespace(cuda=SimpleNamespace(Event=event,current_stream=lambda:'stream'))
    trace=frames.FrameTrace(8,torch);trace.call=15
    with patch.object(frames,'schedstat',side_effect=[[10,20,1],[90,70,3]]),patch.object(frames.time,'perf_counter_ns',side_effect=[100,300]),patch.object(frames.time,'thread_time_ns',side_effect=[100,180]):
        token=trace.begin();trace.end(token)
    assert len(events)==2 and all(e.records==['stream']for e in events)
    with patch.object(frames.logging,'info')as log:
        trace.harvest();assert not log.called
        events[0].complete=True;trace.harvest();assert not log.called
        events[1].complete=True;trace.harvest();data=json.loads(log.call_args.args[1]);sample=data['samples'][0]
    assert sample['forward_host_ms']==.0002 and sample['forward_thread_cpu_ms']==.00008
    assert sample['runqueue_wait_ms']==.00005 and sample['forward_stream_ms']==2.5 and sample['timeslices']==2


def test_backlog_is_bounded_without_waiting():
    torch=SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw:(_ for _ in ()).throw(AssertionError('allocated')),current_stream=lambda:None))
    trace=frames.FrameTrace(0,torch);trace.call=15
    trace.pending=[({},Event(),Event())for _ in range(16)]
    assert trace.begin()is None and len(trace.pending)==16


def test_default_off_installs_no_wrapper():
    model=SimpleNamespace(forward=lambda:123);original=model.forward
    with patch.dict(frames.os.environ,{'BATCHGEN_R1_SM103_DECODE_DIAGNOSTICS':'0'}):frames.install(model,0,None)
    assert model.forward is original
