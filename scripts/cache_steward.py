#!/usr/bin/env python3
"""Retire only pinned EXL3 shard file-cache pages during GB10 model loading.

This is an independent, bounded startup helper, not a system-wide cache flush.
It cannot weaken the separate MemAvailable/NVRM guardian. Stop it after READY.
"""
import argparse
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.check_artifacts import check  # noqa: E402
from scripts.retire_page_cache import MIB, memfree_mib, resident_bytes  # noqa: E402


def container_state(name: str):
    proc = subprocess.run(['docker', 'container', 'inspect', name,
                           '--format', '{{.State.Running}}'],
                          capture_output=True, text=True, timeout=5)
    if proc.returncode == 0:
        return proc.stdout.strip() == 'true'
    if 'No such object' in proc.stderr or 'No such container' in proc.stderr:
        return None
    raise RuntimeError(f'docker inspect failed rc={proc.returncode}: {proc.stderr[:150]}')


def mem_available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    raise RuntimeError('MemAvailable missing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--cache-limit-mib', type=int, default=512)
    a = parser.parse_args()
    if (a.interval < 0.25 or a.interval > 10 or a.max_seconds <= 0 or a.max_seconds > 600
            or a.cache_limit_mib < 64 or a.max_samples < 0):
        raise ValueError('unsafe steward cadence/duration/budget')
    lock = json.loads((PROJECT / 'runtime.lock.json').read_text())
    check(a.target, 'Glm5NextForConditionalGeneration', lock['target_config_sha256'],
          31, index_hash=lock['target_index_sha256'])
    check(a.draft, 'DFlash2DraftModel', lock['draft_config_sha256'], 1)
    paths = [p for root in (a.target, a.draft) for p in sorted(root.glob('*.safetensors'))]
    handles = []
    try:
        for path in paths:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise ValueError(f'non-file shard: {path}')
            handles.append(fd)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with a.out.open('w', buffering=1) as out:
            out.write(json.dumps({'event': 'armed', 'pid': os.getpid(),
                                  'files': len(paths), 'cache_limit_mib': a.cache_limit_mib,
                                  'interval_seconds': a.interval}) + '\n')
            seen_container = False
            iteration = 0
            try:
                while True:
                    now = time.monotonic()
                    if a.stop_file.exists():
                        out.write(json.dumps({'event': 'stopped', 'reason': 'stop.request'}) + '\n')
                        return
                    if now - start >= a.max_seconds:
                        raise RuntimeError('steward duration expired')
                    state = container_state('glm53-exl3')
                    if state is True:
                        seen_container = True
                    elif seen_container:
                        out.write(json.dumps({'event': 'stopped', 'reason': 'container_exited'}) + '\n')
                        return
                    elif now - start >= 120:
                        raise RuntimeError('owned model container never appeared')
                    before = sum(resident_bytes(path) for path in paths)
                    free_before = memfree_mib()
                    used = before > a.cache_limit_mib * MIB
                    if used:
                        for fd in handles:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    after = sum(resident_bytes(path) for path in paths) if used else before
                    out.write(json.dumps({'event': 'sample', 'tick': iteration,
                                          'elapsed_seconds': round(now - start, 2),
                                          'cached_before_mib': before // MIB,
                                          'cached_after_mib': after // MIB,
                                          'memfree_before_mib': free_before,
                                          'memfree_after_mib': memfree_mib(),
                                          'memavailable_mib': mem_available_mib(),
                                          'retired': used, 'container_running': state is True}) + '\n')
                    iteration += 1
                    if a.max_samples and iteration >= a.max_samples:
                        out.write(json.dumps({'event': 'stopped', 'reason': 'max_samples'}) + '\n')
                        return
                    time.sleep(a.interval)
            except Exception as exc:
                out.write(json.dumps({'event': 'failed', 'type': type(exc).__name__,
                                      'message': str(exc)[:180]}) + '\n')
                raise
    finally:
        for fd in handles:
            os.close(fd)


if __name__ == '__main__':
    main()
