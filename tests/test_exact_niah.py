import argparse
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

# Importing this module must stay lightweight: transformers is imported lazily
# inside load_tokenizer so the offline project venv never needs it.
from scripts import exact_niah


class ImportWeightTest(unittest.TestCase):
    def test_transformers_not_imported_at_module_load(self):
        self.assertNotIn('transformers', sys.modules)


class MakeTest(unittest.TestCase):
    def test_single_and_two_key_placement(self):
        single, codes = exact_niah.make(100, 'case-A', False)
        self.assertEqual(codes, ['SKU-case-A'])
        self.assertEqual(single.count('SKU-case-A'), 1)
        multi, codes = exact_niah.make(100, 'case-B', True)
        self.assertEqual(codes, ['SKU-ALPHA-case-B', 'SKU-BRAVO-case-B'])
        self.assertTrue(all(multi.count(x) == 1 for x in codes))
        self.assertLess(multi.index(codes[0]), multi.index(codes[1]))

    def test_deterministic_for_same_inputs(self):
        self.assertEqual(exact_niah.make(30, 't', True), exact_niah.make(30, 't', True))
        self.assertNotEqual(exact_niah.make(30, 't1', False)[0], exact_niah.make(30, 't2', False)[0])


class FitTest(unittest.TestCase):
    def test_token_fit_stays_in_budget(self):
        class Toy:
            def apply_chat_template(self, messages, **kwargs):
                return [0] * len(messages[0]['content'])

        _, _, count = exact_niah.fit(Toy(), 800, 'test-fit', False)
        self.assertTrue(700 <= count <= 800)

    def test_fit_uses_apply_chat_template_with_overhead(self):
        class Toy:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
                assert kwargs.get('reasoning_effort') == 'low'
                return [0] * (len(messages[0]['content']) // 3 + 7)

        _, _, count = exact_niah.fit(Toy(), 400, 'test-fit', False)
        self.assertTrue(300 <= count <= 400)

    def test_unreachable_target_raises(self):
        class Toy:
            def apply_chat_template(self, messages, **kwargs):
                return [0] * (len(messages[0]['content']) + 1_000_000)

        with self.assertRaisesRegex(ValueError, 'token fit'):
            exact_niah.fit(Toy(), 50, 'test-fit', False)

    def test_served_bos_is_counted_only_when_explicitly_admitted(self):
        class Toy:
            def apply_chat_template(self, messages, **kwargs):
                return [154822, 154824, 154828]

        self.assertEqual(exact_niah.count(Toy(), 'x'), 3)
        self.assertEqual(exact_niah.count(Toy(), 'x', bos_overhead=1), 4)


class JobPlanTest(unittest.TestCase):
    def test_four_cases_targets_and_only_filter(self):
        jobs = exact_niah.job_plan(4096, 2048, 'all')
        self.assertEqual([j[0] for j in jobs], ['single25', 'single50', 'single90', 'multi33_66'])
        self.assertEqual([j[3] for j in jobs], [512, 1024, 1843, 1843])
        self.assertEqual([j[0] for j in exact_niah.job_plan(4096, 2048, 'single50')], ['single50'])
        self.assertEqual([j[0] for j in exact_niah.job_plan(4096, 2048, 'multi33_66')], ['multi33_66'])
        self.assertTrue(exact_niah.job_plan(4096, 2048, 'multi33_66')[0][2])

    def test_unknown_case_rejected(self):
        with self.assertRaises(ValueError):
            exact_niah.job_plan(4096, 2048, 'nope')

    def test_near_window_is_an_explicit_single_or_two_key_case(self):
        single = exact_niah.job_plan(262144, 2048, 'single_near')
        multi = exact_niah.job_plan(262144, 2048, 'multi_near')
        self.assertEqual(single, [('single_near', 1.0, False, 260000)])
        self.assertEqual(multi, [('multi_near', 1.0, True, 260000)])


class WindowGuardTest(unittest.TestCase):
    def test_request_must_fit_window_with_reserve(self):
        exact_niah.check_window(2048, 2048, 4096)
        with self.assertRaisesRegex(ValueError, 'window'):
            exact_niah.check_window(2049, 2048, 4096)


class AuditTest(unittest.TestCase):
    def test_api_count_mismatch_is_not_a_pass(self):
        with self.assertRaisesRegex(ValueError, 'token parity'):
            exact_niah.audit(100, {'usage': {'prompt_tokens': 99},
                                   'choices': [{'finish_reason': 'stop',
                                                'message': {'content': 'SKU-case-A'}}]}, ['SKU-case-A'])

    def test_pass_requires_stop_and_needles(self):
        good = {'usage': {'prompt_tokens': 100},
                'choices': [{'finish_reason': 'stop', 'message': {'content': 'Found SKU-case-A.'}}]}
        self.assertEqual(exact_niah.audit(100, good, ['SKU-case-A']), 'Found SKU-case-A.')
        truncated = {'usage': {'prompt_tokens': 100},
                     'choices': [{'finish_reason': 'length', 'message': {'content': 'Found SKU-case-A.'}}]}
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            exact_niah.audit(100, truncated, ['SKU-case-A'])
        miss = {'usage': {'prompt_tokens': 100},
                'choices': [{'finish_reason': 'stop', 'message': {'content': 'no code here'}}]}
        with self.assertRaisesRegex(ValueError, 'needle'):
            exact_niah.audit(100, miss, ['SKU-case-A'])

class BosPolicyTest(unittest.TestCase):
    def test_independent_server_ids_gate_the_exact_bos(self):
        class Toy:
            def apply_chat_template(self, messages, tokenize=True, **kwargs):
                return [154822, 154824, 154828] if tokenize else '[gMASK]<sop><|user|>x'

        good = json.dumps({'length': 4, 'tokens': [1, 154822, 154824, 154828]}).encode()
        bad = json.dumps({'length': 3, 'tokens': [154822, 154824, 154828]}).encode()
        with tempfile.TemporaryDirectory() as td:
            proof = Path(td) / 'bos-proof.json'
            with mock.patch.object(exact_niah.urllib.request, 'urlopen',
                                   return_value=io.BytesIO(good)):
                self.assertEqual(exact_niah.prove_server_bos(Toy(), 'http://endpoint', proof), 1)
            saved = json.loads(proof.read_text())
            self.assertEqual((saved['local_count'], saved['server_count'], saved['inserted_bos_id']),
                             (3, 4, 1))
            with mock.patch.object(exact_niah.urllib.request, 'urlopen',
                                   return_value=io.BytesIO(bad)):
                with self.assertRaisesRegex(ValueError, 'BOS'):
                    exact_niah.prove_server_bos(Toy(), 'http://endpoint', proof)


class PayloadTest(unittest.TestCase):
    def test_payload_contract(self):
        p = exact_niah.payload_for('target', 'long prompt', 2048)
        self.assertEqual(p['model'], 'target')
        self.assertEqual(p['messages'], [{'role': 'user', 'content': 'long prompt'}])
        self.assertEqual(p['temperature'], 0)
        self.assertEqual(p['max_tokens'], 2048)
        self.assertEqual(p['chat_template_kwargs'], {'reasoning_effort': 'low'})


CODE_RE = re.compile(r'SKU-[A-Za-z0-9_-]+')


class ToyTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        return [0] * (len(messages[0]['content']) // 3 + 7)


def fake_ask(base, model, prompt, reserve):
    ids = ToyTokenizer().apply_chat_template([{'role': 'user', 'content': prompt}])
    codes = sorted(set(CODE_RE.findall(prompt)))
    return {'usage': {'prompt_tokens': len(ids) + 1, 'completion_tokens': 5},
            'choices': [{'finish_reason': 'stop', 'message': {'content': 'answer: ' + ' '.join(codes)}}]}


def run_main(tmp, window, only):
    out = Path(tmp) / f'{only}.json'
    args = argparse.Namespace(target='/models/target', base='http://endpoint', model='target',
                              window=window, only=only, out=out)
    printed = io.StringIO()
    with mock.patch.object(exact_niah.argparse.ArgumentParser, 'parse_args', return_value=args), \
            mock.patch.object(exact_niah, 'load_tokenizer', return_value=ToyTokenizer()), \
            mock.patch.object(exact_niah, 'prove_server_bos', return_value=1), \
            mock.patch.object(exact_niah, 'ask', side_effect=fake_ask), redirect_stdout(printed):
        exact_niah.main()
    return json.loads(out.read_text()), printed.getvalue()


class MainFlowTest(unittest.TestCase):
    def test_all_four_cases_report_local_api_parity(self):
        with tempfile.TemporaryDirectory() as td:
            data, out = run_main(td, 4096, 'all')
        self.assertIn('NIAH_EXACT_PASS window=4096 rows=4', out)
        self.assertEqual([r['case'] for r in data['rows']],
                         ['single25', 'single50', 'single90', 'multi33_66'])
        for row in data['rows']:
            self.assertEqual(row['prompt_tokens'], row['api_prompt_tokens'])
            self.assertEqual(len(row['prompt_sha256']), 64)
            self.assertEqual(row['finish'], 'stop')
            for code in row['codes']:
                self.assertIn(code, row['answer'])
        multi = data['rows'][-1]
        self.assertEqual(len(multi['codes']), 2)

    def test_only_filter_runs_single_case(self):
        with tempfile.TemporaryDirectory() as td:
            data, out = run_main(td, 4096, 'single50')
        self.assertIn('NIAH_EXACT_PASS window=4096 rows=1', out)
        self.assertEqual([r['case'] for r in data['rows']], ['single50'])
