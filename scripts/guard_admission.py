"""Require a fresh, live, fail-closed independent host guard before launch."""

import argparse
import json
import time
from pathlib import Path

_TERMINAL = {'stop_failed', 'stopped', 'expired', 'self_check_failed', 'admission_failed'}


def admit(path: Path, now: float | None = None) -> int:
    if now is None:
        now = time.time()
    if not path.is_file():
        raise ValueError('guard evidence missing')
    try:
        entries = [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, ValueError) as exc:
        raise ValueError('guard evidence unreadable') from exc
    if not entries or entries[0].get('event') != 'armed' or entries[0].get('name') != 'glm53-exl3':
        raise ValueError('guard not armed for owned container')
    if any(entry.get('event') in _TERMINAL for entry in entries):
        raise ValueError('guard terminal event present')
    samples = [entry for entry in entries if 'ts' in entry]
    if not samples:
        raise ValueError('guard has no live sample')
    recent = samples[-1]
    if type(recent.get('ts')) not in (int, float) or not 0 <= now - recent['ts'] <= 8:
        raise ValueError('guard sample stale or from future')
    memory = recent.get('available_mib')
    if type(memory) is not int or memory < 6144 or recent.get('reason') is not None:
        raise ValueError('guard reports unsafe host')
    return memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    print(f'GUARD_OK available_mib={admit(args.path)}')
