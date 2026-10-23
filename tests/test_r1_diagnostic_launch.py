"""CPU-only launch-plan contract checks; no torch or runtime imports."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r1_diagnostic_launch', ROOT / 'scripts/r1_reproduction/launch_r1_diagnostic.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class DiagnosticLaunchTest(unittest.TestCase):
    def test_both_nodes_same_topology_and_direct_converted_directory(self):
        for rank in (0, 1):
            env, cmd = launch.plan(launch.ROOT, rank, 'contract-v1')
            get = lambda key: cmd[cmd.index(key) + 1]
            self.assertEqual(get('--converted-ckpt-dir'), str(launch.ROOT / 'converted'))
            self.assertEqual(get('--model'), 'deepseek-ai/DeepSeek-R1')
            self.assertEqual((get('--world-size'), get('--nnodes'), get('--node-rank')), ('16', '2', str(rank)))
            self.assertEqual(get('--host-kv-cache-size'), '64')
            self.assertEqual(get('--input-ids-pool-size-gb'), '1')
            self.assertEqual(get('--kv-dtype'), 'bfloat16')
            self.assertIn('--disable-cuda-graphs', cmd)
            for flag in ('--enable-cuda-graph', '--enable-prefix-cache', '--enable-hugetlbfs', '--fast-init', '--enable-ep-with-offloading'):
                self.assertNotIn(flag, cmd)
            self.assertEqual(env['BATCHGEN_R1_SM103_ATTENTION'], 'triton_precision')
            self.assertEqual(env['BATCHGEN_W8A16_DEQUANT'], '0')
            self.assertEqual(env['BATCHGEN_ENABLE_ALL_TO_ALL'], '0')
            self.assertEqual(env['GLOO_SOCKET_IFNAME'], 'eth0')

    def test_only_current_http_cli_arguments(self):
        tree = ast.parse((ROOT / 'batchgen/server/server_args.py').read_text())
        accepted = {arg.value for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
        _, cmd = launch.plan(launch.ROOT, 0, 'contract-v1')
        self.assertTrue(set(token for token in cmd if token.startswith('--')) <= accepted)


if __name__ == '__main__':
    unittest.main()
