import json
import tempfile
import time
import unittest
from pathlib import Path

from scripts.guard_admission import admit

ROOT = Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    def test_owned_single_gpu_loopback_envelope(self):
        text = (ROOT / 'scripts/serve.sh').read_text()
        for required in ('--gpus all', '--cpus=14', '--memory=112g', '--memory-swap=112g',
                         '127.0.0.1:5013:5000', ':/models/target:ro', ':/models/draft:ro',
                         'check_artifacts.py', 'guard_admission.py', 'tmux -L',
                         'has-session', 'docker container inspect'):
            self.assertIn(required, text)
        self.assertIn('guard_admission.py" "$GUARD_LOG" >&2', text)
        self.assertIn('--draft "$DRAFT" >&2', text)
        self.assertIn('retire_page_cache.py', text)
        self.assertLess(text.index('check_artifacts.py'), text.index('retire_page_cache.py'))
        self.assertLess(text.index('retire_page_cache.py'), text.index('exec docker run -d'))
        self.assertIn('--out "$CACHE_RETIRE_RECEIPT" >&2', text)

    def test_guard_must_be_live_after_armed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'guard.jsonl'
            now = time.time()
            armed = {'event': 'armed', 'since': 'test', 'name': 'glm53-exl3'}
            sample = {'ts': now, 'available_mib': 9000, 'reason': None}
            path.write_text(json.dumps(armed) + '\n' + json.dumps(sample) + '\n')
            self.assertEqual(admit(path, now=now), 9000)
            with self.assertRaisesRegex(ValueError, 'stale'):
                admit(path, now=now + 11)
            path.write_text(path.read_text() + json.dumps({'event': 'stop_failed'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'terminal'):
                admit(path, now=now)

    def test_armed_alone_is_not_sufficient(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'guard.jsonl'
            path.write_text(json.dumps({'event': 'armed', 'name': 'glm53-exl3'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'sample'):
                admit(path, now=time.time())
