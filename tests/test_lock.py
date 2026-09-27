import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class LockTest(unittest.TestCase):
    def test_exact_native_source(self):
        x = json.loads((ROOT / 'runtime.lock.json').read_text())
        self.assertEqual(x['exllamav3_sha'], '12414d0af7b3beeabdda5990f6b554b996fa1416')
        self.assertEqual(x['tabbyapi_sha'], '816c32195887aaecea1c64528f2921566766259b')
        self.assertEqual(x['torch'], '2.13.0+cu130')
        self.assertEqual(x['exllamav3_version'], '1.5.2')
        self.assertEqual(x['target_index_sha256'], '9287eb16ff7b9e549eba6a12957a6fd218ef407ebbcc84403c92f912aa3bb0df')
        self.assertEqual(x['platform'], 'linux/arm64-sm121')
        self.assertNotIn('weights', x)
