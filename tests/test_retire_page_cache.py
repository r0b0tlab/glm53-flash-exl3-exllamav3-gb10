import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.retire_page_cache import admit_budget, resident_bytes, retire

MIB = 1048576


class PageCacheRetirementTest(unittest.TestCase):
    def test_only_local_shard_cache_is_retired(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shard = root / 'one.safetensors'
            shard.write_bytes(b'a' * 8192)
            with mock.patch('scripts.retire_page_cache.resident_bytes', side_effect=[4096, 0]), \
                 mock.patch('scripts.retire_page_cache.os.posix_fadvise') as advise:
                result = retire(root, expected=1)
            self.assertEqual(result['before_bytes'], 4096)
            self.assertEqual(result['after_bytes'], 0)
            self.assertEqual(result['total_bytes'], 8192)
            advise.assert_called_once()
            self.assertEqual(shard.read_bytes(), b'a' * 8192)

    def test_wrong_count_and_symlinks_are_rejected_before_fadvise(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'one.safetensors').write_bytes(b'a')
            with mock.patch('scripts.retire_page_cache.os.posix_fadvise') as advise:
                with self.assertRaisesRegex(ValueError, 'expected 2'):
                    retire(root, expected=2)
                alias = root / 'alias.safetensors'
                alias.symlink_to(root / 'one.safetensors')
                with self.assertRaisesRegex(ValueError, 'unsafe shard'):
                    retire(root, expected=2)
                advise.assert_not_called()
            link = root.parent / (root.name + '-link')
            link.symlink_to(root, target_is_directory=True)
            try:
                with self.assertRaisesRegex(ValueError, 'unsafe model root'):
                    retire(link, expected=2)
            finally:
                link.unlink()

    def test_fail_closed_on_retained_pages_and_low_physical_free(self):
        self.assertEqual(admit_budget(100 * MIB, 0, 5000)['required_memfree_mib'], 4196)
        with self.assertRaisesRegex(ValueError, 'cache residency'):
            admit_budget(100 * MIB, 65 * MIB, 5000)
        with self.assertRaisesRegex(ValueError, 'physical free'):
            admit_budget(100 * MIB, 0, 4095)

    def test_mincore_reports_bounded_residency(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'sample.safetensors'
            p.write_bytes(b'a' * 8192)
            self.assertTrue(0 <= resident_bytes(p) <= p.stat().st_size)
