"""Fail-closed model-shard page-cache retirement for GB10 unified memory.

Linux MemAvailable counts clean file-cache pages that the NVIDIA GB10 driver
cannot reliably reclaim for a native device allocation. On this host a warm
~42 GiB EXL3 shard cache coincided with NV_ERR_NO_MEMORY at 59 GiB available;
retiring ONLY these read-only shard pages raised MemFree from ~59 to ~101 GiB
and the same guarded K5 profile then loaded and served. Never drop host-wide
caches, touch model bytes, relax the independent guard, or ignore a failed
mincore readback.
"""

import argparse
import ctypes
import json
import os
from datetime import datetime
from pathlib import Path

PAGE = os.sysconf('SC_PAGE_SIZE')
MIB = 1048576
CACHE_LIMIT = 64 * MIB
WORKSPACE_RESERVE_MIB = 4096
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mmap.restype = ctypes.c_void_p
LIBC.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                     ctypes.c_int, ctypes.c_int, ctypes.c_longlong)
LIBC.mincore.restype = ctypes.c_int
LIBC.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
LIBC.munmap.restype = ctypes.c_int
LIBC.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)


def memfree_mib() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemFree:'):
            return int(line.split()[1]) // 1024
    raise RuntimeError('MemFree missing')


def resident_bytes(path: Path) -> int:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f'empty shard: {path}')
    pages = (size + PAGE - 1) // PAGE
    bitmap = ctypes.create_string_buffer(pages)
    with path.open('rb', buffering=0) as handle:
        address = LIBC.mmap(None, size, 1, 1, handle.fileno(), 0)
        if address in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), 'mmap', str(path))
        try:
            if LIBC.mincore(address, size, bitmap):
                raise OSError(ctypes.get_errno(), 'mincore', str(path))
            count = sum(value & 1 for value in bitmap.raw[:pages])
        finally:
            if LIBC.munmap(address, size):
                raise OSError(ctypes.get_errno(), 'munmap', str(path))
    return min(size, count * PAGE)


def retire(root: Path, expected: int) -> dict:
    root = root.absolute()
    if not root.is_dir() or root != root.resolve(strict=True):
        raise ValueError(f'unsafe model root: {root}')
    files = sorted(root.glob('*.safetensors'))
    if len(files) != expected:
        raise ValueError(f'expected {expected} shards in {root}; got {len(files)}')
    for path in files:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f'unsafe shard: {path}')
    before = sum(resident_bytes(path) for path in files)
    for path in files:
        with path.open('rb', buffering=0) as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    after = sum(resident_bytes(path) for path in files)
    return {'root': str(root), 'files': len(files),
            'total_bytes': sum(path.stat().st_size for path in files),
            'before_bytes': before, 'after_bytes': after}


def admit_budget(total_bytes: int, remaining_bytes: int, free_mib: int) -> dict:
    if remaining_bytes > CACHE_LIMIT:
        raise ValueError(f'cache residency remains {remaining_bytes // MIB} MiB (>64 MiB)')
    required = (total_bytes + MIB - 1) // MIB + WORKSPACE_RESERVE_MIB
    if free_mib < required:
        raise ValueError(f'physical free {free_mib} MiB below {required} MiB model+workspace budget')
    return {'required_memfree_mib': required, 'remaining_cached_mib': remaining_bytes // MIB}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    before_free = memfree_mib()
    target = retire(args.target, 31)
    draft = retire(args.draft, 1)
    after_free = memfree_mib()
    total = target['total_bytes'] + draft['total_bytes']
    remaining = target['after_bytes'] + draft['after_bytes']
    receipt = {'timestamp': datetime.now().astimezone().isoformat(),
               'target': target, 'draft': draft, 'memfree_before_mib': before_free,
               'memfree_after_mib': after_free}
    try:
        receipt['budget'] = admit_budget(total, remaining, after_free)
        receipt['status'] = 'PASS'
    except ValueError as exc:
        receipt['status'] = 'BLOCKED'
        receipt['reason'] = str(exc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    if receipt['status'] != 'PASS':
        raise RuntimeError(f'model cache/physical memory admission failed: {receipt["reason"]}')
    print(f'CACHE_RETIRE_OK files=32 cached_before_mib={(target["before_bytes"]+draft["before_bytes"])//MIB} '
          f'cached_after_mib={remaining//MIB} memfree_mib={after_free} '
          f'required_mib={receipt["budget"]["required_memfree_mib"]}')


if __name__ == '__main__':
    main()
