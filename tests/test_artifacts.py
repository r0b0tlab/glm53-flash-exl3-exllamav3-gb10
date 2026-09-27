import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.check_artifacts import check


class ArtifactsTest(unittest.TestCase):
    def make_pack(self, root: Path):
        blob = json.dumps({'architectures': ['Glm5NextForConditionalGeneration']}).encode()
        (root / 'config.json').write_bytes(blob)
        (root / 'quantization_config.json').write_text('{"quant_method":"exl3"}')
        (root / 'model.safetensors.index.json').write_text(json.dumps({
            'metadata': {'total_size': 7},
            'weight_map': {'x': 'model.safetensors'},
        }))
        header = json.dumps({'x': {'dtype': 'U8', 'shape': [7], 'data_offsets': [0, 7]}}).encode()
        shard = struct.pack('<Q', len(header)) + header + b'fixture'
        return hashlib.sha256(blob).hexdigest(), shard

    def test_missing_indexed_shard(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            digest, shard = self.make_pack(p)
            with self.assertRaisesRegex(ValueError, 'missing shard'):
                check(p, 'Glm5NextForConditionalGeneration', digest, 1)
            (p / 'model.safetensors').write_bytes(shard)
            self.assertEqual(check(p, 'Glm5NextForConditionalGeneration', digest, 1), 1)

    def test_truncated_or_appended_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            digest, shard = self.make_pack(p)
            (p / 'model.safetensors').write_bytes(shard[:-1])
            with self.assertRaisesRegex(ValueError, 'shard size'):
                check(p, 'Glm5NextForConditionalGeneration', digest, 1)
            (p / 'model.safetensors').write_bytes(shard + b'X')
            with self.assertRaisesRegex(ValueError, 'shard size'):
                check(p, 'Glm5NextForConditionalGeneration', digest, 1)

    def test_unindexed_extra_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            digest, shard = self.make_pack(p)
            (p / 'model.safetensors').write_bytes(shard)
            (p / 'unexpected.safetensors').write_bytes(shard)
            with self.assertRaisesRegex(ValueError, 'unindexed shard'):
                check(p, 'Glm5NextForConditionalGeneration', digest, 1)

    def test_symlinked_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            real = p / 'real'
            real.mkdir()
            digest, shard = self.make_pack(real)
            (real / 'model.safetensors').write_bytes(shard)
            alias = p / 'alias'
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'unsafe model root'):
                check(alias, 'Glm5NextForConditionalGeneration', digest, 1)

    def test_index_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            digest, shard = self.make_pack(p)
            (p / 'model.safetensors').write_bytes(shard)
            with self.assertRaisesRegex(ValueError, 'index identity'):
                check(p, 'Glm5NextForConditionalGeneration', digest, 1, index_hash='0' * 64)
