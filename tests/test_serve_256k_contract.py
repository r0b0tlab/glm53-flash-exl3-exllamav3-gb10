"""Offline contracts for the single-owner, guarded 256K serve entrypoint."""
import re
import subprocess
import unittest
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/serve_256k.sh'
PROFILE = ROOT / 'config/gb10-256k-k5.yml'


class Serve256KContractTest(unittest.TestCase):
    def test_pinned_profile_matches_the_proven_one_variable_trial(self):
        p = YAML(typ='safe').load(PROFILE.read_text())
        self.assertEqual((p['model']['max_seq_len'], p['model']['cache_size'],
                          p['model']['max_batch_size'], p['model']['cache_mode']),
                         (262144, 262144, 1, 'FP16'))
        self.assertEqual(p['memory']['sysmem_recurrent_cache'], 512)
        self.assertEqual(p['draft_model']['draft_num_tokens'], 5)
        self.assertEqual(p['draft_model']['draft_mode'], 'model')
        self.assertTrue(p['model']['vision'])
        self.assertEqual(p['network']['host'], '0.0.0.0')
        self.assertTrue(p['network']['disable_fetch_requests'])

    def test_owner_orders_guard_steward_load_and_stops_steward_on_ready(self):
        s = SCRIPT.read_text()
        for needle in ('git status --porcelain=v1', 'source-revision.txt',
                       'image-inspect.json', 'scripts/guard_admission.py',
                       'scripts/cache_steward.py', 'STEWARD_PID', 'STEWARD_STOP',
                       'scripts/serve.sh', '/v1/model', 'TARGET_256K_READY',
                       'steward-stop-proof.txt', 'docker stop --timeout 3',
                       'guard-live.txt', 'stop.request',
                       "--image-id-file '$OUT/image-id.txt'"):
            self.assertIn(needle, s)
        self.assertLess(s.index("out.joinpath('image-id.txt')"),
                        s.index('tmux -L glm53-exl3-guard new-session'))
        self.assertLess(s.index('tmux -L glm53-exl3-guard new-session'),
                        s.index('tmux -L glm53-exl3-cache new-session'))
        self.assertLess(s.index('tmux -L glm53-exl3-cache new-session'),
                        s.index('bash scripts/serve.sh'))
        self.assertLess(s.index("printf '%s\\n' 'TARGET_256K_READY'"),
                        s.index('steward-stop-proof.txt'))
        self.assertIn('GUARD_LEASE_SECONDS=86400', s)
        self.assertIn('$(date +%s) -ge $((START_EPOCH+GUARD_LEASE_SECONDS-100))', s)
        self.assertIn('sysmem_recurrent_cache', s)
        self.assertNotIn('drop_caches', s)
        self.assertNotIn('docker rm', s)
        self.assertNotIn('0.0.0.0:5013:', s)

    def test_shell_and_every_embedded_python_block_compile(self):
        subprocess.run(['bash', '-n', str(SCRIPT)], check=True, capture_output=True)
        blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?=\n|$)", SCRIPT.read_text(), re.S)
        self.assertGreaterEqual(len(blocks), 9)
        for i, block in enumerate(blocks):
            compile(block, f'{SCRIPT}:embedded_{i}', 'exec')


if __name__ == '__main__':
    unittest.main()
