"""Real CPython GC contracts run in children to preserve the test runner's GC."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def child(body):
    source_path = os.environ.get('R1_GC_SOURCE', str(ROOT/'batchgen/worker/r1_decode_gc.py'))
    source = """
import gc, importlib.util, json, logging, os, sys, time, types, weakref
from pathlib import Path
spec=importlib.util.spec_from_file_location('scope',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert gc.get_freeze_count()==0
class Node: pass
""".replace('SOURCE', repr(source_path))
    r = subprocess.run([sys.executable, '-c', source + textwrap.dedent(body)],
                       capture_output=True, text=True, timeout=15)
    assert r.returncode == 0, r.stdout + r.stderr


def test_frozen_cycle_young_cycles_refcount_and_phase_release():
    child('''
    stable=Node();stable.cycle=stable;old=weakref.ref(stable)
    scope=m.DecodeGCScope(0);scope.activate();assert scope.owns_freeze and gc.isenabled()
    del stable;gc.collect();assert old()is not None
    young=Node();young.cycle=young;young_ref=weakref.ref(young)
    del young;gc.collect();assert young_ref()is None
    plain=Node();plain_ref=weakref.ref(plain);del plain;assert plain_ref()is None
    assert scope.stats[2]['collections']>=3
    scope.close();scope.close();gc.collect()
    assert old()is None and gc.get_freeze_count()==0 and gc.isenabled()
    assert scope.callback not in gc.callbacks
    ''')


def test_existing_freeze_and_disabled_collection_keep_their_owner():
    child('''
    stable=Node();stable.cycle=stable;old=weakref.ref(stable);gc.freeze()
    scope=m.DecodeGCScope(0);scope.activate();assert not scope.owns_freeze
    scope.close();del stable;gc.collect();assert old()is not None and gc.get_freeze_count()>0
    gc.unfreeze();gc.collect();assert old()is None
    gc.disable();scope=m.DecodeGCScope(0);scope.activate();scope.close()
    assert not gc.isenabled() and gc.get_freeze_count()==0
    gc.enable()
    ''')


def test_opt_in_success_exception_and_unsupported_lifetime():
    child('''
    class Model:
        def forward(self,x):return x+1
    model=Model();before=list(gc.callbacks)
    os.environ['BATCHGEN_R1_SM103_FREEZE_DECODE_GC']='0';m.install(model,0)
    assert not hasattr(model,'_r1_decode_gc_scope')and gc.callbacks==before
    os.environ.update(BATCHGEN_R1_SM103_FREEZE_DECODE_GC='1',BATCHGEN_R1_SM103_FAST_DIAGNOSTIC='1',BATCHGEN_R1_SM103_ALLOW_CAPTURE='0')
    try:m.install(model,0,persistent_phase_instances=True)
    except ValueError:pass
    else:raise AssertionError('Persistent model lifetime accepted')
    assert gc.callbacks==before
    m.install(model,0);assert model.forward(2)==3 and model.forward(7)==8
    scope=model._r1_decode_gc_scope;assert scope.owns_freeze and scope.stats[2]['collections']>=1
    scope.close();assert gc.get_freeze_count()==0 and gc.callbacks==before
    class Broken:
        def forward(self):raise RuntimeError('original forward failure')
    broken=Broken();m.install(broken,0)
    try:broken.forward()
    except RuntimeError:pass
    else:raise AssertionError('Original error suppressed')
    assert not broken._r1_decode_gc_scope.attempted
    broken._r1_decode_gc_scope.close();assert gc.callbacks==before
    ''')


def test_actual_worker_teardown_unfreezes_before_dropping_both_model_owners():
    external = os.environ.get('R1_GC_WORKER_METHOD')
    if external:
        method = Path(external).read_text()
    else:
        tree = ast.parse((ROOT/'batchgen/batchgen_worker.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='deep_free_model_memory')
        method = ast.unparse(node)
    child(textwrap.dedent('''
    os.environ.update(BATCHGEN_R1_SM103_FREEZE_DECODE_GC='1',BATCHGEN_R1_SM103_FAST_DIAGNOSTIC='1',BATCHGEN_R1_SM103_ALLOW_CAPTURE='0')
    class Model:
        def forward(self,x):return x
    model=Model();m.install(model,0);model.forward(1)
    ref=weakref.ref(model);scope=model._r1_decode_gc_scope
    class Worker:
        rank=0
        def _log_phase_memory_ledger(self,stage):pass
    worker=Worker();worker.model=model
    worker.parallel_manager=types.SimpleNamespace(model=model,persistent_phase_instances=False)
    torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
    METHOD
    worker.deep_free_model_memory=types.MethodType(deep_free_model_memory,worker)
    del model;worker.deep_free_model_memory();gc.collect()
    assert worker.model is None and worker.parallel_manager.model is None
    assert ref()is None and scope.closed and gc.get_freeze_count()==0 and gc.isenabled()
    ''').replace('METHOD', method))
