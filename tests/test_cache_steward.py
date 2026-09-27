import argparse
import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import cache_steward

MIB = 1048576


class ContainerIdentityTest(unittest.TestCase):
    def test_absent_is_distinct_from_docker_failure(self):
        def result(rc, err='', out=''):
            return mock.Mock(returncode=rc, stderr=err, stdout=out)
        with mock.patch.object(cache_steward.subprocess, 'run', return_value=result(1, 'No such object: glm53-exl3')):
            self.assertIsNone(cache_steward.container_state('glm53-exl3'))
        with mock.patch.object(cache_steward.subprocess, 'run', return_value=result(1, 'Cannot connect to daemon')):
            with self.assertRaisesRegex(RuntimeError, 'docker inspect failed'):
                cache_steward.container_state('glm53-exl3')
        with mock.patch.object(cache_steward.subprocess, 'run', return_value=result(0, out='true\n')):
            self.assertTrue(cache_steward.container_state('glm53-exl3'))


class StewardLifecycleTest(unittest.TestCase):
    def fixture(self, root):
        target, draft = root / 'target', root / 'draft'
        target.mkdir()
        draft.mkdir()
        (target / 'one.safetensors').write_bytes(b'target' * 1024)
        (draft / 'two.safetensors').write_bytes(b'draft' * 1024)
        (root / 'runtime.lock.json').write_text(json.dumps({
            'target_config_sha256': 'a' * 64, 'target_index_sha256': 'b' * 64,
            'draft_config_sha256': 'c' * 64}))
        return argparse.Namespace(target=target, draft=draft, out=root / 'steward.jsonl',
                                  stop_file=root / 'stop.request', interval=0.25,
                                  max_seconds=30, max_samples=2, cache_limit_mib=64)

    def test_bounded_two_sample_run_retires_only_pinned_fds_and_closes_them(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self.fixture(root)
            with mock.patch.object(cache_steward, 'PROJECT', root), \
                    mock.patch.object(cache_steward.argparse.ArgumentParser, 'parse_args', return_value=args), \
                    mock.patch.object(cache_steward, 'check') as check, \
                    mock.patch.object(cache_steward, 'container_state', side_effect=[None, True]), \
                    mock.patch.object(cache_steward, 'resident_bytes',
                                      side_effect=[40*MIB, 40*MIB, 0, 0]*2), \
                    mock.patch.object(cache_steward, 'memfree_mib', return_value=100000), \
                    mock.patch.object(cache_steward, 'mem_available_mib', return_value=6500), \
                    mock.patch.object(cache_steward.os, 'posix_fadvise') as advise, \
                    mock.patch.object(cache_steward.time, 'sleep'):
                cache_steward.main()
            self.assertEqual(check.call_count, 2)
            self.assertEqual(advise.call_count, 4)
            for call in advise.call_args_list:
                with self.assertRaises(OSError) as err:
                    os.fstat(call.args[0])
                self.assertEqual(err.exception.errno, errno.EBADF)
            rows = [json.loads(line) for line in args.out.read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows],
                             ['armed', 'sample', 'sample', 'stopped'])
            self.assertEqual(rows[-1]['reason'], 'max_samples')
            self.assertTrue(all(row['retired'] for row in rows[1:3]))
            self.assertTrue((args.target / 'one.safetensors').read_bytes())
            self.assertTrue((args.draft / 'two.safetensors').read_bytes())

    def test_artifact_rejection_precedes_all_model_fd_activity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self.fixture(root)
            with mock.patch.object(cache_steward, 'PROJECT', root), \
                    mock.patch.object(cache_steward.argparse.ArgumentParser, 'parse_args', return_value=args), \
                    mock.patch.object(cache_steward, 'check', side_effect=ValueError('not pinned')) as check, \
                    mock.patch.object(cache_steward.os, 'posix_fadvise') as advise:
                with self.assertRaisesRegex(ValueError, 'not pinned'):
                    cache_steward.main()
            check.assert_called_once()
            advise.assert_not_called()
            self.assertFalse(args.out.exists())

    def test_symlink_is_rejected_even_after_artifact_checker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self.fixture(root)
            (args.draft / 'two.safetensors').unlink()
            (args.draft / 'two.safetensors').symlink_to(args.target / 'one.safetensors')
            with mock.patch.object(cache_steward, 'PROJECT', root), \
                    mock.patch.object(cache_steward.argparse.ArgumentParser, 'parse_args', return_value=args), \
                    mock.patch.object(cache_steward, 'check'), \
                    mock.patch.object(cache_steward.os, 'posix_fadvise') as advise:
                with self.assertRaises(OSError):
                    cache_steward.main()
            advise.assert_not_called()
            self.assertFalse(args.out.exists())
