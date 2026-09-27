"""Independent host-memory and privileged-kernel guard for the owned container."""

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import time

NAME = 'glm53-exl3'
_OOM = re.compile(r'NV_ERR_NO_MEMORY|oom-kill|Killed process')
_CONTAINER_ID = re.compile(r'^[0-9a-f]{64}$')
_IMAGE_ID = re.compile(r'^sha256:[0-9a-f]{64}$')


def unsafe(available_mib: int, kernel_oom: bool, floor: int = 6144):
    if kernel_oom:
        return 'kernel_oom'
    if available_mib < floor:
        return 'low_memory'
    return None


def mem_available():
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    raise RuntimeError('MemAvailable missing')


def parse_journal(returncode: int, stdout: str, stderr: str) -> bool:
    if stderr.strip() or returncode not in (0, 1):
        raise RuntimeError('privileged kernel journal unavailable')
    if returncode == 1:
        if stdout.strip() not in ('', '-- No entries --'):
            raise RuntimeError('privileged kernel journal unexpected empty-result output')
        return False
    if not _OOM.search(stdout):
        raise RuntimeError('privileged kernel journal returned empty or non-OOM output')
    return True


def new_oom(start):
    result = subprocess.run(
        ['sudo', '-n', 'journalctl', '-b', '-k', '--since', start, '--no-pager',
         '--grep', _OOM.pattern, '-n', '1'],
        capture_output=True, text=True, timeout=8,
    )
    return parse_journal(result.returncode, result.stdout, result.stderr)


def stop_owned(expected_image_id=None):
    # Snapshot the named container's immutable image before stopping its exact
    # ID. Never pass a mutable name to docker stop after an ownership check.
    proc = subprocess.run(['docker', 'container', 'inspect', NAME, '--format',
                           '{{.Id}} {{.Image}} {{.State.Running}}'], check=True,
                          capture_output=True, text=True, timeout=10)
    fields = proc.stdout.strip().split()
    if (len(fields) != 3 or not _CONTAINER_ID.fullmatch(fields[0]) or
            not _IMAGE_ID.fullmatch(fields[1]) or fields[2] not in ('true', 'false')):
        raise ValueError('unverifiable owned container identity')
    if expected_image_id is not None and fields[1] != expected_image_id:
        raise ValueError('image identity mismatch; refusing to stop foreign container')
    if fields[2] == 'true':
        subprocess.run(['docker', 'stop', '--time', '3', fields[0]],
                       check=True, timeout=15)


def _stop_and_record(out, reason: str, event: str, status: int,
                     expected_image_id=None) -> int:
    try:
        stop_owned(expected_image_id)
    except Exception as exc:
        out.write(json.dumps({'event': 'stop_failed', 'reason': reason,
                              'error_type': type(exc).__name__, 'name': NAME}) + '\n')
        return 4
    out.write(json.dumps({'event': event, 'reason': reason, 'name': NAME}) + '\n')
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--duration-seconds', type=int, default=7200)
    parser.add_argument('--image-id-file', type=pathlib.Path)
    args = parser.parse_args()
    output = args.out
    duration = getattr(args, 'duration_seconds', 7200)
    if type(duration) is not int or not 1 <= duration <= 86400:
        raise ValueError('guard lease must be 1..86400 seconds')
    image_file = getattr(args, 'image_id_file', None)
    expected_image_id = None if image_file is None else image_file.read_text().strip()
    if expected_image_id is not None and not _IMAGE_ID.fullmatch(expected_image_id):
        raise ValueError('invalid pinned image identity')
    start = datetime.datetime.now().astimezone().isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', buffering=1) as out:
        try:
            available = mem_available()
            reason = unsafe(available, new_oom(start))
        except Exception as exc:
            out.write(json.dumps({'event': 'self_check_failed', 'reason': type(exc).__name__}) + '\n')
            return 5
        if reason:
            out.write(json.dumps({'event': 'admission_failed', 'reason': reason,
                                  'available_mib': available}) + '\n')
            return 5
        out.write(json.dumps({'event': 'armed', 'since': start, 'name': NAME,
                              'lease_seconds': duration,
                              'expected_image_id': expected_image_id}) + '\n')
        end = time.monotonic() + duration
        while time.monotonic() < end:
            available = None
            try:
                available = mem_available()
                reason = unsafe(available, new_oom(start))
            except Exception as exc:
                reason = 'guard_probe_failed:' + type(exc).__name__
            out.write(json.dumps({'ts': time.time(), 'available_mib': available,
                                  'reason': reason}) + '\n')
            if reason:
                return _stop_and_record(out, reason, 'stopped', 2, expected_image_id)
            time.sleep(2)
        return _stop_and_record(out, 'guard_expired', 'expired', 3, expected_image_id)


if __name__ == '__main__':
    raise SystemExit(main())
