import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ImageContractTest(unittest.TestCase):
    def test_native_arm64_pins_and_nonroot(self):
        lock = json.loads((ROOT / 'runtime.lock.json').read_text())
        dockerfile = (ROOT / 'docker/Dockerfile').read_text()
        for pin in (lock['cuda_image'], lock['exllamav3_sha'], lock['tabbyapi_sha'],
                    lock['torch']):
            self.assertIn(pin, dockerfile)
        self.assertIn('TORCH_CUDA_ARCH_LIST="12.1"', dockerfile)
        self.assertNotIn('FROM --platform=', dockerfile)  # build command pins --platform; Dockerfile stays portable to BuildKit
        self.assertIn('pip install --no-build-isolation', dockerfile)
        self.assertIn('USER runner', dockerfile)
        self.assertIn('NATIVE_EXTENSION_PRESENT', dockerfile)
        self.assertIn('COPY scripts/smoke.py', dockerfile)
        self.assertIn('COPY scripts/repair_cusparselt_wheel.py', dockerfile)
        self.assertIn('python /opt/repair_cusparselt_wheel.py', dockerfile)
        self.assertLess(dockerfile.index('pip install --no-build-isolation'),
                        dockerfile.index('COPY scripts/repair_cusparselt_wheel.py'))
        self.assertIn('python -m pip check', dockerfile)
        self.assertIn('uvloop==0.22.1', dockerfile)
        self.assertLess(dockerfile.index('git clone https://github.com/theroyallab/tabbyAPI.git'),
                        dockerfile.index('uvloop==0.22.1'))
        self.assertIn('import uvloop; uvloop.install()', dockerfile)
        self.assertIn('WORKDIR /opt/tabbyAPI', dockerfile)
        self.assertNotIn('EXLLAMA_NOCOMPILE', dockerfile)
        self.assertNotIn('.[cu13]', dockerfile)

    def test_project_revision_is_required(self):
        dockerfile = (ROOT / 'docker/Dockerfile').read_text()
        self.assertIn('ARG PROJECT_REVISION', dockerfile)
        self.assertIn('test -n "$PROJECT_REVISION"', dockerfile)
        self.assertIn('org.opencontainers.image.revision', dockerfile)
