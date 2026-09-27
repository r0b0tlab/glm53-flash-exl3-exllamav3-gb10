"""Fail-closed, read-only admission for the pinned local EXL3 safetensors packs."""

import argparse
import hashlib
import json
from pathlib import Path

_MAX_HEADER_BYTES = 64 * 1024 * 1024


def check_shard(path: Path) -> None:
    """Validate a safetensors header's offsets against the actual file length.

    Index metadata.total_size is *not* a trustworthy exact file-byte count: the
    local target's advertised tensor total differs from its valid headers by 8192.
    Every shard is checked independently without reading its weight payload.
    """
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f'shard size/header invalid: {path}')
        header_size = int.from_bytes(prefix, 'little')
        if header_size < 2 or header_size > _MAX_HEADER_BYTES or size <= 8 + header_size:
            raise ValueError(f'shard size/header invalid: {path}')
        try:
            header = json.loads(stream.read(header_size))
            if not isinstance(header, dict):
                raise ValueError('header must be an object')
            regions = sorted(tuple(value['data_offsets']) for key, value in header.items()
                             if key != '__metadata__')
            if not regions:
                raise ValueError('no tensor offsets')
            end = 0
            for start, stop in regions:
                if type(start) is not int or type(stop) is not int or start != end or stop < start:
                    raise ValueError('noncontiguous tensor offsets')
                end = stop
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'shard size/header invalid: {path}') from exc
    if end != size - 8 - header_size:
        raise ValueError(f'shard size mismatch: {path}')


def check(root: Path, arch: str, expected_hash: str, shards: int,
          index_hash: str | None = None) -> int:
    root = root.absolute()
    if not root.is_dir() or root != root.resolve(strict=True):
        raise ValueError(f'unsafe model root: {root}')
    cfg = root / 'config.json'
    if (not cfg.is_file() or cfg.is_symlink()
            or hashlib.sha256(cfg.read_bytes()).hexdigest() != expected_hash):
        raise ValueError(f'config identity mismatch: {root}')
    if json.loads(cfg.read_text()).get('architectures') != [arch]:
        raise ValueError(f'architecture mismatch: {root}')
    quant_cfg = root / 'quantization_config.json'
    if not quant_cfg.is_file() or quant_cfg.is_symlink():
        raise ValueError(f'not EXL3: {root}')
    if json.loads(quant_cfg.read_text()).get('quant_method') != 'exl3':
        raise ValueError(f'not EXL3: {root}')
    idx = root / 'model.safetensors.index.json'
    if idx.exists():
        if idx.is_symlink() or (index_hash is not None
                               and hashlib.sha256(idx.read_bytes()).hexdigest() != index_hash):
            raise ValueError(f'index identity mismatch: {root}')
        names = sorted(set(json.loads(idx.read_text())['weight_map'].values()))
        actual = sorted(p.name for p in root.glob('*.safetensors'))
        if actual != names:
            raise ValueError(f'unindexed shard or missing shard: {root}')
    else:
        if index_hash is not None:
            raise ValueError(f'index identity mismatch: {root}')
        names = sorted(p.name for p in root.glob('*.safetensors'))
    if (len(names) != shards or any(Path(n).name != n or (root / n).is_symlink()
                                   or not (root / n).is_file() for n in names)):
        raise ValueError(f'missing shard or unsafe index: {root}')
    for name in names:
        check_shard(root / name)
    return len(names)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--draft', type=Path, required=True)
    a = p.parse_args()
    lock = json.loads((Path(__file__).resolve().parents[1] / 'runtime.lock.json').read_text())
    t = check(a.target, 'Glm5NextForConditionalGeneration', lock['target_config_sha256'],
              31, index_hash=lock['target_index_sha256'])
    d = check(a.draft, 'DFlash2DraftModel', lock['draft_config_sha256'], 1)
    print(f'ARTIFACTS_OK target={t} draft={d}')
