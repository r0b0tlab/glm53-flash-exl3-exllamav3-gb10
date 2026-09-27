import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import guard


class GuardTest(unittest.TestCase):
    def test_bound_image_stops_only_exact_owned_container_id(self):
        container_id = 'a' * 64
        image_id = 'sha256:' + 'b' * 64
        inspect = mock.Mock(returncode=0, stdout=f'{container_id} {image_id} true\n', stderr='')
        with mock.patch('scripts.guard.subprocess.run', side_effect=[inspect, mock.Mock(returncode=0)]) as run:
            guard.stop_owned(image_id)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0],
                         ['docker', 'stop', '--time', '3', container_id])

    def test_image_mismatch_never_stops_foreign_named_container(self):
        inspect = mock.Mock(returncode=0,
                            stdout=f"{'a'*64} sha256:{'c'*64} true\n", stderr='')
        with mock.patch('scripts.guard.subprocess.run', return_value=inspect) as run:
            with self.assertRaisesRegex(ValueError, 'image identity'):
                guard.stop_owned('sha256:' + 'b'*64)
        run.assert_called_once()

    def test_guard_accepts_only_a_valid_pinned_image_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            owner_image = root / 'image-id.txt'
            owner_image.write_text('sha256:' + 'b'*64 + '\n')
            args = argparse.Namespace(out=root/'guard.jsonl', duration_seconds=10,
                                      image_id_file=owner_image)
            with mock.patch('scripts.guard.argparse.ArgumentParser.parse_args', return_value=args), \
                    mock.patch('scripts.guard.mem_available', return_value=9000), \
                    mock.patch('scripts.guard.new_oom', side_effect=[False, RuntimeError('probe failed')]), \
                    mock.patch('scripts.guard.stop_owned') as stop:
                self.assertEqual(guard.main(), 2)
            stop.assert_called_once_with('sha256:'+'b'*64)
            self.assertEqual(json.loads(args.out.read_text().splitlines()[0])['expected_image_id'],
                             'sha256:'+'b'*64)
            owner_image.write_text('not-an-image\n')
            with mock.patch('scripts.guard.argparse.ArgumentParser.parse_args', return_value=args), \
                    mock.patch('scripts.guard.new_oom') as journal:
                with self.assertRaisesRegex(ValueError, 'image identity'):
                    guard.main()
            journal.assert_not_called()

    def test_memory_and_kernel_triggers(self):
        self.assertEqual(guard.unsafe(6143, False), 'low_memory')
        self.assertEqual(guard.unsafe(20000, True), 'kernel_oom')
        self.assertIsNone(guard.unsafe(6144, False))

    def test_journal_parser_fail_closed(self):
        self.assertTrue(guard.parse_journal(0, 'NVRM: NV_ERR_NO_MEMORY\n', ''))
        self.assertFalse(guard.parse_journal(1, '-- No entries --\n', ''))
        for code, stdout, stderr in ((0, '', ''), (0, 'x', 'Permission denied'),
                                     (2, '', 'read error')):
            with self.subTest(code=code, stdout=stdout, stderr=stderr):
                with self.assertRaisesRegex(RuntimeError, 'journal'):
                    guard.parse_journal(code, stdout, stderr)

    def test_meminfo_failure_after_arm_stops_owned_container(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'guard.jsonl'
            args = argparse.Namespace(out=out)
            with mock.patch('scripts.guard.argparse.ArgumentParser.parse_args', return_value=args), \
                 mock.patch('scripts.guard.mem_available', side_effect=[9000, OSError('ENOMEM')]), \
                 mock.patch('scripts.guard.new_oom', return_value=False), \
                 mock.patch('scripts.guard.stop_owned') as stop:
                self.assertEqual(guard.main(), 2)
            stop.assert_called_once()
            events = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(events[0]['event'], 'armed')
            self.assertEqual(events[-1]['event'], 'stopped')
            self.assertIn('guard_probe_failed', events[-1]['reason'])

    def test_bad_journal_is_not_armed(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'guard.jsonl'
            args = argparse.Namespace(out=out)
            with mock.patch('scripts.guard.argparse.ArgumentParser.parse_args', return_value=args), \
                 mock.patch('scripts.guard.mem_available', return_value=9000), \
                 mock.patch('scripts.guard.new_oom', side_effect=RuntimeError('denied')), \
                 mock.patch('scripts.guard.stop_owned') as stop:
                self.assertNotEqual(guard.main(), 0)
            stop.assert_not_called()
            events = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertFalse(any(x.get('event') == 'armed' for x in events))
            self.assertEqual(events[-1]['event'], 'self_check_failed')

    def test_missing_meminfo_key_fails(self):
        with mock.patch('scripts.guard.pathlib.Path.read_text', return_value='MemTotal: 32768 kB\n'):
            with self.assertRaisesRegex(RuntimeError, 'MemAvailable missing'):
                guard.mem_available()
        with mock.patch('scripts.guard.pathlib.Path.read_text', return_value='MemAvailable: not-a-number kB\n'):
            with self.assertRaises(ValueError):
                guard.mem_available()

    def test_low_memory_is_not_armed(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'guard.jsonl'
            args = argparse.Namespace(out=out)
            with mock.patch('scripts.guard.argparse.ArgumentParser.parse_args', return_value=args), \
                 mock.patch('scripts.guard.mem_available', return_value=6143), \
                 mock.patch('scripts.guard.new_oom', return_value=False), \
                 mock.patch('scripts.guard.stop_owned') as stop:
                self.assertEqual(guard.main(), 5)
            stop.assert_not_called()
            events = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(events[-1]['event'], 'admission_failed')
            self.assertFalse(any(x.get('event') == 'armed' for x in events))

    def test_journal_failure_after_arm_stops_owned_container(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'guard.jsonl'
            args = argparse.Namespace(out=out)
            with mock.patch('scripts.guard.argparse.ArgumentParser.parse_args', return_value=args), \
                 mock.patch('scripts.guard.mem_available', return_value=9000), \
                 mock.patch('scripts.guard.new_oom', side_effect=[False, RuntimeError('denied')]), \
                 mock.patch('scripts.guard.stop_owned') as stop:
                self.assertEqual(guard.main(), 2)
            stop.assert_called_once()
            self.assertIn('guard_probe_failed', json.loads(out.read_text().splitlines()[-1])['reason'])

    def test_failed_stop_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'guard.jsonl'
            args = argparse.Namespace(out=out)
            with mock.patch('scripts.guard.argparse.ArgumentParser.parse_args', return_value=args), \
                 mock.patch('scripts.guard.mem_available', side_effect=[9000, 1000]), \
                 mock.patch('scripts.guard.new_oom', return_value=False), \
                 mock.patch('scripts.guard.stop_owned', side_effect=subprocess.TimeoutExpired(['docker','stop'],15)):
                self.assertEqual(guard.main(), 4)
            events = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(events[-1]['event'], 'stop_failed')

    def test_explicit_24h_guard_lease_expires_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'guard.jsonl'
            args = argparse.Namespace(out=out, duration_seconds=86400)
            with mock.patch('scripts.guard.argparse.ArgumentParser.parse_args', return_value=args), \
                    mock.patch('scripts.guard.mem_available', return_value=9000), \
                    mock.patch('scripts.guard.new_oom', return_value=False), \
                    mock.patch('scripts.guard.time.monotonic', side_effect=[0, 86401]), \
                    mock.patch('scripts.guard.stop_owned') as stop:
                self.assertEqual(guard.main(), 3)
            stop.assert_called_once()
            events = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(events[0]['lease_seconds'], 86400)
            self.assertEqual((events[-1]['event'], events[-1]['reason']),
                             ('expired', 'guard_expired'))

    def test_invalid_guard_lease_is_rejected_before_probing(self):
        for lease in (0, 86401):
            with self.subTest(lease=lease), tempfile.TemporaryDirectory() as td:
                args = argparse.Namespace(out=Path(td) / 'guard.jsonl', duration_seconds=lease)
                with mock.patch('scripts.guard.argparse.ArgumentParser.parse_args', return_value=args), \
                        mock.patch('scripts.guard.mem_available') as mem, \
                        mock.patch('scripts.guard.new_oom') as journal, \
                        mock.patch('scripts.guard.stop_owned') as stop:
                    with self.assertRaisesRegex(ValueError, 'lease'):
                        guard.main()
                mem.assert_not_called()
                journal.assert_not_called()
                stop.assert_not_called()
