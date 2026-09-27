"""Offline tests for the exact-source, fail-closed ARM64 build receipt."""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/build_image.sh'


class BuildImageContractTest(unittest.TestCase):
    def test_clean_main_and_host_admission_precedes_native_build(self):
        s = SCRIPT.read_text()
        for check in ('git status --porcelain=v1', 'git branch --show-current',
                      'docker ps -q', 'nvidia-smi --query-compute-apps=pid',
                      'MemAvailable', 'source-revision.txt', 'source-tree.txt',
                      'build.rc', 'build-pass.timestamp', 'image-inspect.json',
                      'docker buildx build --load --platform linux/arm64',
                      'PROJECT_REVISION=$REV', 'org.opencontainers.image.revision',
                      'scripts/cache_steward.py', 'scripts/serve_256k.sh',
                      'config/gb10-256k-k5.yml'):
            self.assertIn(check, s)
        self.assertLess(s.index('git status --porcelain=v1'),
                        s.index('docker buildx build --load --platform linux/arm64'))
        self.assertLess(s.index('nvidia-smi --query-compute-apps=pid'),
                        s.index('docker buildx build --load --platform linux/arm64'))
        self.assertNotIn('docker builder prune', s)
        self.assertNotIn('docker system prune', s)

    def test_shell_and_embedded_python_blocks_compile(self):
        subprocess.run(['bash', '-n', str(SCRIPT)], check=True, capture_output=True)
        blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY(?=\n|$)", SCRIPT.read_text(), re.S)
        self.assertEqual(len(blocks), 1)
        compile(blocks[0], f'{SCRIPT}:image_audit', 'exec')


if __name__ == '__main__':
    unittest.main()
