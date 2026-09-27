import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import bench

STRUCTURED = ('Output a JSON array of 25 countries. For each, an object with keys "country", "capital", '
              '"population_millions" (number, 1 decimal). Output only the JSON, no markdown fences, no commentary.')
CODE = ('Write a complete Python class RingBuffer with methods push, pop, is_empty, and __len__, backed by a '
        'fixed-size list with head/tail indices. Include type hints and a short docstring per method. '
        'Output only the code.')
PROSE = ('Write a short story, about 200 words, about a lighthouse keeper who finds a message in a bottle. '
         'Use varied prose.')

import re

CONCURRENCY_KEYS = re.compile(r'^c\d+$')


class CompletionTokensTest(unittest.TestCase):
    def test_missing_count_rejected(self):
        with self.assertRaises(ValueError):
            bench.completion_tokens({'usage': {}, 'choices': [{'finish_reason': 'stop'}]})
        with self.assertRaises(ValueError):
            bench.completion_tokens({'usage': {'completion_tokens': 0}, 'choices': [{'finish_reason': 'stop'}]})
        with self.assertRaises(ValueError):
            bench.completion_tokens({'usage': {'completion_tokens': -3}, 'choices': [{'finish_reason': 'stop'}]})
        with self.assertRaises(ValueError):
            bench.completion_tokens({'usage': {'completion_tokens': True}, 'choices': [{'finish_reason': 'stop'}]})
        with self.assertRaises(ValueError):
            bench.completion_tokens({'usage': {'completion_tokens': 17}, 'choices': []})
        self.assertEqual(
            bench.completion_tokens({'usage': {'completion_tokens': 17}, 'choices': [{'finish_reason': 'stop'}]}), 17)


class ExactPromptsTest(unittest.TestCase):
    def test_class_prompts_are_verbatim(self):
        self.assertEqual(list(bench.PROMPTS), ['structured', 'code', 'prose'])
        self.assertEqual(bench.PROMPTS['structured'], STRUCTURED)
        self.assertEqual(bench.PROMPTS['code'], CODE)
        self.assertEqual(bench.PROMPTS['prose'], PROSE)


class SpecCountersTest(unittest.TestCase):
    def test_parse_accepted_and_rejected(self):
        data = {'usage': {'completion_tokens_details': {
            'accepted_prediction_tokens': 9, 'rejected_prediction_tokens': 2}}}
        self.assertEqual(bench.spec_counters(data), {'accepted': 9, 'rejected': 2})

    def test_missing_or_fake_counters_rejected(self):
        for bad in ({'usage': {}},
                    {'usage': {'completion_tokens_details': None}},
                    {'usage': {'completion_tokens_details': {}}},
                    {'usage': {'completion_tokens_details': {'accepted_prediction_tokens': 'many'}}},
                    {'usage': {'completion_tokens_details': {'accepted_prediction_tokens': -1}}}):
            with self.assertRaises(ValueError):
                bench.spec_counters(bad)


class LadderTest(unittest.TestCase):
    def test_concurrency_only_up_to_configured_max(self):
        self.assertEqual(bench.ladder(1), [1])
        self.assertEqual(bench.ladder(3), [1, 2])
        self.assertEqual(bench.ladder(4), [1, 2, 4])
        self.assertEqual(bench.ladder(16), [1, 2, 4, 8, 16])
        self.assertEqual(bench.ladder(0), [])


class PayloadTest(unittest.TestCase):
    def test_unique_salt_rides_in_content(self):
        a = bench.payload_for('m', 'p', 256, 'run-1')
        b = bench.payload_for('m', 'p', 256, 'run-2')
        self.assertIn('\n\nRun id: run-1', a['messages'][0]['content'])
        self.assertIn('\n\nRun id: run-2', b['messages'][0]['content'])
        self.assertEqual(a['temperature'], 0)
        self.assertEqual(a['max_tokens'], 256)
        self.assertEqual(a['chat_template_kwargs'], {'reasoning_effort': 'low'})


def canned_request(calls, accepted, rejected=0):
    def fake(base, model, prompt, cap, salt, mode_has_draft=False):
        calls.append((prompt, cap, salt))
        tokens = max(1, cap // 2)
        return {'tokens': tokens, 'seconds': 0.5, 'tok_s': tokens / 0.5,
                'finish': 'stop', 'accepted': accepted, 'rejected': rejected}

    return fake


def run_main(tmp, mode, max_concurrency, accepted):
    calls = []
    out = Path(tmp) / f'{mode}.json'
    args = argparse.Namespace(base='http://endpoint', model='m', mode=mode,
                              max_concurrency=max_concurrency, out=out)
    with mock.patch.object(bench.argparse.ArgumentParser, 'parse_args', return_value=args), \
            mock.patch.object(bench, 'request', side_effect=canned_request(calls, accepted)):
        bench.main()
    return calls, json.loads(out.read_text())


class MainRunTest(unittest.TestCase):
    def test_spec_run_full_plan_unique_salts_recorded_counters(self):
        with tempfile.TemporaryDirectory() as td:
            calls, data = run_main(td, 'spec', 2, accepted=4)
        self.assertEqual(data['mode'], 'spec')
        self.assertEqual(data['model'], 'm')
        self.assertEqual(data['max_concurrency'], 2)
        self.assertEqual(sorted(k for k in data['rows'] if CONCURRENCY_KEYS.match(k)), ['c1', 'c2'])
        for label in ('structured', 'code', 'prose'):
            self.assertEqual(len(data['rows'][label]['samples']), 5)
            self.assertEqual(data['rows'][label]['finish_counts'], {'stop': 5})
        self.assertEqual(len(data['rows']['c1']['samples']), 1)
        self.assertEqual(len(data['rows']['c2']['samples']), 2)
        self.assertEqual(data['spec_counters'], {'accepted_total': 72, 'rejected_total': 0})
        self.assertEqual(len(calls), 19)  # warmup + 3x5 class + c1 + c2
        salts = [salt for _, _, salt in calls]
        self.assertEqual(len(salts), len(set(salts)))
        self.assertEqual(calls[0], ('Say hello.', 256, 'spec-warmup'))
        self.assertEqual(calls[1], (STRUCTURED, 2048, 'structured-0'))
        self.assertEqual(calls[16], (STRUCTURED, 256, 'spec-c1-0'))

    def test_ar_mode_records_zero_draft_counters_without_gate(self):
        with tempfile.TemporaryDirectory() as td:
            calls, data = run_main(td, 'ar', 1, accepted=0)
        self.assertEqual(data['mode'], 'ar')
        self.assertEqual(data['spec_counters'], {'accepted_total': 0, 'rejected_total': 0})
        self.assertEqual(sorted(k for k in data['rows'] if CONCURRENCY_KEYS.match(k)), ['c1'])

    def test_spec_mode_with_zero_accepted_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'accepted'):
                run_main(td, 'spec', 1, accepted=0)
