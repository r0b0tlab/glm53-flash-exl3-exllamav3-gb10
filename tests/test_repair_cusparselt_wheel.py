"""Regression for NVIDIA's AArch64 cuSPARSELt wheel internal sbsa tag."""
import base64
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.repair_cusparselt_wheel import repair


def record_line(wheel: Path, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
    return f'{wheel.parent.name}/WHEEL,sha256={digest},{len(data)}\n'


class RepairWheelTest(unittest.TestCase):
    def make_fixture(self, root: Path, machine: int = 183):
        info = root / 'nvidia_cusparselt_cu13-0.8.1.dist-info'
        info.mkdir()
        wheel = info / 'WHEEL'
        before = (b'Wheel-Version: 1.0\nGenerator: setuptools (80.9.0)\n'
                  b'Root-Is-Purelib: true\nTag: py3-none-manylinux2014_sbsa\n')
        wheel.write_bytes(before)
        record = info / 'RECORD'
        record.write_text(record_line(wheel, before) + f'{info.name}/RECORD,,\n')
        library = root / 'nvidia/cusparselt/lib/libcusparseLt.so.0'
        library.parent.mkdir(parents=True)
        header = bytearray(64)
        header[:4] = b'\x7fELF'
        header[4] = 2
        header[5] = 1
        header[18:20] = machine.to_bytes(2, 'little')
        library.write_bytes(header)
        return wheel, record, library

    def test_repair_exact_vendor_tag_and_update_record(self):
        with tempfile.TemporaryDirectory() as td:
            wheel, record, library = self.make_fixture(Path(td))
            self.assertEqual(repair(wheel, record, library), 'fixed')
            data = wheel.read_bytes()
            self.assertIn(b'Tag: py3-none-manylinux2014_aarch64\n', data)
            self.assertNotIn(b'manylinux2014_sbsa', data)
            self.assertIn(record_line(wheel, data), record.read_text())
            self.assertEqual(repair(wheel, record, library), 'already-correct')

    def test_reject_wrong_binary_and_tampered_record(self):
        with tempfile.TemporaryDirectory() as td:
            wheel, record, library = self.make_fixture(Path(td), machine=62)
            with self.assertRaisesRegex(ValueError, 'AArch64'):
                repair(wheel, record, library)
            self.assertIn(b'manylinux2014_sbsa', wheel.read_bytes())
        with tempfile.TemporaryDirectory() as td:
            wheel, record, library = self.make_fixture(Path(td))
            record.write_text(record.read_text().replace('sha256=', 'sha256=bad'))
            with self.assertRaisesRegex(ValueError, 'RECORD'):
                repair(wheel, record, library)
            self.assertIn(b'manylinux2014_sbsa', wheel.read_bytes())

    def test_reject_unexpected_tag(self):
        with tempfile.TemporaryDirectory() as td:
            wheel, record, library = self.make_fixture(Path(td))
            data = wheel.read_bytes().replace(b'manylinux2014_sbsa', b'manylinux2014_x86_64')
            wheel.write_bytes(data)
            record.write_text(record_line(wheel, data) + f'{wheel.parent.name}/RECORD,,\n')
            with self.assertRaisesRegex(ValueError, 'unexpected wheel tag'):
                repair(wheel, record, library)
