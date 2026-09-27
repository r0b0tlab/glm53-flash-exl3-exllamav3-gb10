import unittest
from pathlib import Path
from ruamel.yaml import YAML

class ProfileTest(unittest.TestCase):
    def test_single_gpu_private_dflash(self):
        x = YAML(typ='safe').load((Path(__file__).resolve().parents[1]/'config/config.yml').read_text())
        self.assertEqual((x['network']['host'], x['network']['port']), ('0.0.0.0', 5000))
        self.assertTrue(x['network']['disable_fetch_requests'])
        self.assertEqual(x['network']['allowed_origins'], [])
        self.assertEqual(x['model']['cache_size'], x['model']['max_seq_len'])
        self.assertEqual(x['model']['max_batch_size'], 1)
        self.assertEqual(x['draft_model']['draft_mode'], 'model')
        self.assertEqual(x['draft_model']['draft_model_name'], 'draft')
