import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('qa_guard', SCRIPTS/'run_qa_sanity_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class QAGuardAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.manifest = root/'manifest.json'
        self.manifest.write_text('{}')
        self.admission = root/'admission.json'
        self.values = {'scope': guard.qa.SCOPE, 'manifest_sha256': guard.qa.sha(self.manifest),
            'server_identity': 'sglang0.5.5.post3+R1-routing-compat',
            'cohort': 'sglang-r1-diagnostic-v4', 'endpoint': guard.ENDPOINT,
            'kv_dtype': 'torch.bfloat16', 'model_id': 'deepseek-ai/DeepSeek-R1'}
        self.admission.write_text(json.dumps(self.values))
        self.args = SimpleNamespace(manifest=self.manifest, manifest_sha=guard.qa.sha(self.manifest),
            qa_source_sha=guard.qa.sha(guard.qa.__file__), admission=self.admission, output=root/'new-run')

    def test_cpu_plan_does_not_launch_a_client(self):
        with patch.object(guard.qa, 'client_plan', return_value={'command': ['python3', 'client.py']}), \
             patch.object(guard.subprocess, 'Popen') as popen:
            _, plan = guard.verified_plan(self.args)
        popen.assert_not_called()
        self.assertEqual(plan['command'][0], str(guard.ROOT/'client-env/bin/python'))
        self.assertFalse(plan['quality_pass'])
        self.assertFalse(self.args.output.exists())

    def test_stale_manifest_and_source_rejected_before_request_plan(self):
        for field in ('manifest_sha', 'qa_source_sha'):
            original = getattr(self.args, field)
            setattr(self.args, field, '0'*64)
            with patch.object(guard.qa, 'client_plan') as client:
                with self.assertRaises(ValueError):
                    guard.verified_plan(self.args)
                client.assert_not_called()
            setattr(self.args, field, original)

    def test_incompatible_deployment_rejected(self):
        for field in ('scope', 'manifest_sha256', 'server_identity', 'cohort', 'endpoint', 'kv_dtype', 'model_id'):
            with self.subTest(field=field):
                self.admission.write_text(json.dumps({**self.values, field: 'changed'}))
                with patch.object(guard.qa, 'client_plan') as client:
                    with self.assertRaises(ValueError):
                        guard.verified_plan(self.args)
                    client.assert_not_called()

    def test_same_pid_with_changed_start_ticks_is_not_the_service(self):
        saved = {'pid': 123, 'pgid': 123, 'session': 123, 'start_ticks': 42}
        for current in (None, {**saved, 'start_ticks': 43, 'state': 'R'}, {**saved, 'state': 'Z'}):
            with patch.object(guard.ownership, 'process_identity', return_value=current):
                self.assertFalse(guard.service_current({'service_process_identity': saved}))
        with patch.object(guard.ownership, 'process_identity', return_value={**saved, 'state': 'S'}):
            self.assertTrue(guard.service_current({'service_process_identity': saved}))

    def test_wrong_container_cannot_create_outputs_or_launch(self):
        with patch.object(guard.socket, 'gethostname', return_value='different-host'), \
             patch.object(guard.subprocess, 'Popen') as popen:
            with self.assertRaises(RuntimeError):
                guard.execute(self.args, self.values, {})
        popen.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_cleanup_defers_repeated_stop_and_restores_handlers_on_error(self):
        original = object()
        with patch.object(guard.signal, 'getsignal', return_value=original), \
             patch.object(guard.signal, 'signal') as change, \
             patch.object(guard.ownership, 'stop_own_group', side_effect=RuntimeError('cleanup failure')):
            with self.assertRaises(RuntimeError):
                guard.cleanup_client(object(), {})
        self.assertEqual(change.call_args_list[0].args, (guard.signal.SIGTERM, guard.signal.SIG_IGN))
        self.assertEqual(change.call_args_list[1].args, (guard.signal.SIGINT, guard.signal.SIG_IGN))
        self.assertEqual(change.call_args_list[2].args, (guard.signal.SIGTERM, original))
        self.assertEqual(change.call_args_list[3].args, (guard.signal.SIGINT, original))
