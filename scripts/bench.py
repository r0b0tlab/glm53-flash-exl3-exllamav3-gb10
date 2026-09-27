"""Fail-closed matched HTTP performance client (Task 8).

E2E tokens over request/full-round wall time only. Usage counters are the
single source of truth: a missing completion-token count or a spec run whose
accepted draft tokens never exceed zero is a hard error, never a fallback.
"""
import argparse
import concurrent.futures
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

CONCURRENCY_KEY = re.compile(r'^c\d+$')

PROMPTS = {
    'structured': 'Output a JSON array of 25 countries. For each, an object with keys "country", "capital", '
                  '"population_millions" (number, 1 decimal). Output only the JSON, no markdown fences, no commentary.',
    'code': 'Write a complete Python class RingBuffer with methods push, pop, is_empty, and __len__, backed by a '
            'fixed-size list with head/tail indices. Include type hints and a short docstring per method. '
            'Output only the code.',
    'prose': 'Write a short story, about 200 words, about a lighthouse keeper who finds a message in a bottle. '
             'Use varied prose.'}

SALTS_PER_LEVEL = 5
TIMEOUT = 1200


def completion_tokens(data):
    n = (data.get('usage') or {}).get('completion_tokens')
    if type(n) is not int or n <= 0 or not data.get('choices'):
        raise ValueError('missing real completion-token count')
    return n


def spec_counters(data):
    details = (data.get('usage') or {}).get('completion_tokens_details') or {}
    accepted = details.get('accepted_prediction_tokens')
    rejected = details.get('rejected_prediction_tokens')
    if type(accepted) is not int or type(rejected) is not int or accepted < 0 or rejected < 0:
        raise ValueError('missing real draft acceptance/rejection counters')
    return {'accepted': accepted, 'rejected': rejected}


def payload_for(model, prompt, cap, salt):
    return {'model': model,
            'messages': [{'role': 'user', 'content': prompt + '\n\nRun id: ' + salt}],
            'temperature': 0,
            'max_tokens': cap,
            'reasoning_effort': 'low',
            'chat_template_kwargs': {'reasoning_effort': 'low'}}


def request(base, model, prompt, cap, salt, mode_has_draft=False):
    payload = payload_for(model, prompt, cap, salt)
    req = urllib.request.Request(base + '/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as f:
        data = json.load(f)
    seconds = time.perf_counter() - started
    tokens = completion_tokens(data)
    choice = data['choices'][0]
    if choice.get('finish_reason') not in ('stop', 'length'):
        raise ValueError(f'unexpected finish_reason: {choice.get("finish_reason")!r}')
    if not ((choice.get('message') or {}).get('content') or (choice.get('message') or {}).get('reasoning_content')):
        raise ValueError('empty output')
    row = {'tokens': tokens, 'seconds': seconds, 'tok_s': tokens / seconds,
           'finish': choice.get('finish_reason')}
    try:
        row.update(spec_counters(data))
    except ValueError:
        if mode_has_draft:
            raise
        row.update(accepted=0, rejected=0)
    return row


def ladder(max_concurrency):
    return [c for c in (1, 2, 4, 8, 16) if c <= max_concurrency]


def run_class(base, model, label, prompt, mode_has_draft=False):
    samples = [request(base, model, prompt, 2048, f'{label}-{i}', mode_has_draft)
               for i in range(SALTS_PER_LEVEL)]
    counts = {}
    for sample in samples:
        counts[sample['finish']] = counts.get(sample['finish'], 0) + 1
    return {'median_tok_s': statistics.median(s['tok_s'] for s in samples),
            'finish_counts': counts,
            'samples': samples}


def run_concurrency(base, model, concurrency, salt_prefix, mode_has_draft=False):
    prompts = [PROMPTS['structured']] * concurrency
    caps = [256] * concurrency
    salts = [f'{salt_prefix}-{i}' for i in range(concurrency)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(lambda i: request(base, model, prompts[i], caps[i], salts[i], mode_has_draft),
                                range(concurrency)))
    wall = time.perf_counter() - started
    return {'aggregate_tok_s': sum(s['tokens'] for s in samples) / wall,
            'wall_seconds': wall,
            'samples': samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:5013')
    parser.add_argument('--model', default='target')
    parser.add_argument('--mode', choices=('ar', 'spec'), required=True)
    parser.add_argument('--max-concurrency', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    has_draft = args.mode == 'spec'
    request(args.base, args.model, 'Say hello.', 256, args.mode + '-warmup', has_draft)
    rows = {label: run_class(base=args.base, model=args.model, label=label, prompt=prompt,
                             mode_has_draft=has_draft)
            for label, prompt in PROMPTS.items()}
    for c in ladder(args.max_concurrency):
        rows[f'c{c}'] = run_concurrency(args.base, args.model, c, f'{args.mode}-c{c}', has_draft)
    accepted_total = sum(s['accepted'] for row in rows.values() for s in row['samples'])
    rejected_total = sum(s['rejected'] for row in rows.values() for s in row['samples'])
    if has_draft and accepted_total <= 0:
        raise ValueError('spec mode reported zero accepted draft tokens; not a real speculative pass')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'mode': args.mode, 'model': args.model,
                                    'max_concurrency': args.max_concurrency,
                                    'spec_counters': {'accepted_total': accepted_total,
                                                      'rejected_total': rejected_total},
                                    'rows': rows}, indent=2) + '\n')
    print(f'BENCH_OK mode={args.mode} classes=3 max_concurrency={args.max_concurrency}')


if __name__ == '__main__':
    main()
