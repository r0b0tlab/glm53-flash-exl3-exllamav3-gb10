"""Correct NVIDIA cuSPARSELt 0.8.1's mislabelled ARM64 WHEEL metadata.

NVIDIA publishes an AArch64 wheel filename and AArch64 ELF payload, but its
internal WHEEL tag is `manylinux2014_sbsa`, which pip's AArch64 platform tag
set rejects. Patch only this exact installed artifact and update RECORD; never
suppress `pip check`, delete the CUDA dependency, or alter compiled libraries.
"""

import base64
import hashlib
import importlib.metadata
import os
import platform
import sys
import tempfile
from pathlib import Path

PACKAGE = 'nvidia-cusparselt-cu13'
DIST_INFO = 'nvidia_cusparselt_cu13-0.8.1.dist-info'
OLD = b'Tag: py3-none-manylinux2014_sbsa\n'
NEW = b'Tag: py3-none-manylinux2014_aarch64\n'
LIBRARY = 'nvidia/cusparselt/lib/libcusparseLt.so.0'


def _entry(wheel: Path, contents: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b'=').decode()
    return f'{wheel.parent.name}/WHEEL,sha256={digest},{len(contents)}\n'


def _atomic_replace(path: Path, contents: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.wheel-tag-', delete=False) as temp:
        candidate = Path(temp.name)
        temp.write(contents)
    try:
        candidate.chmod(path.stat().st_mode & 0o777)
        os.replace(candidate, path)
    finally:
        candidate.unlink(missing_ok=True)


def repair(wheel: Path, record: Path, library: Path) -> str:
    if sys.platform != 'linux' or platform.machine() != 'aarch64':
        raise ValueError('AArch64 Linux platform required')
    if wheel.parent.name != DIST_INFO or record != wheel.parent / 'RECORD':
        raise ValueError('unexpected NVIDIA cuSPARSELt distribution path')
    with library.open('rb') as f:
        header = f.read(20)
    if (len(header) != 20 or header[:4] != b'\x7fELF' or header[4:6] != b'\x02\x01'
            or int.from_bytes(header[18:20], 'little') != 183):
        raise ValueError('cuSPARSELt shared library is not ELF AArch64')
    source = wheel.read_bytes()
    if source.count(OLD) == 1 and source.count(b'Tag: ') == 1:
        fixed = source.replace(OLD, NEW)
        state = 'fixed'
    elif source.count(NEW) == 1 and source.count(b'Tag: ') == 1:
        fixed = source
        state = 'already-correct'
    else:
        raise ValueError('unexpected wheel tag; do not rewrite an unverified distribution')
    lines = record.read_text().splitlines(keepends=True)
    key = f'{DIST_INFO}/WHEEL,'
    matches = [(i, line) for i, line in enumerate(lines) if line.startswith(key)]
    if len(matches) != 1 or matches[0][1] != _entry(wheel, source):
        raise ValueError('RECORD hash/size mismatch before wheel tag repair')
    if state == 'fixed':
        _atomic_replace(wheel, fixed)
        lines[matches[0][0]] = _entry(wheel, fixed)
        _atomic_replace(record, ''.join(lines).encode())
    return state


def main() -> None:
    dist = importlib.metadata.distribution(PACKAGE)
    if dist.version != '0.8.1':
        raise ValueError(f'unexpected cuSPARSELt version: {dist.version}')
    wheel = Path(str(dist.locate_file(f'{DIST_INFO}/WHEEL')))
    library = Path(str(dist.locate_file(LIBRARY)))
    state = repair(wheel, wheel.parent / 'RECORD', library)
    print(f'CUSPARSELT_WHEEL_TAG_{state.upper().replace("-", "_")} {wheel}')


if __name__ == '__main__':
    main()
