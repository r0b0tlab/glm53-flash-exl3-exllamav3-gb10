"""Actual-token NIAH client (Task 12a).

Every case is fitted with the real local tokenizer via
``apply_chat_template`` (never a chars/token guess), audited against the
API's ``usage.prompt_tokens`` (exact parity required). TabbyAPI's effective
GLM tokenizer inserts one BOS token before the chat template. Never assume
that prefix: prove it independently against /v1/token/encode for this epoch
before fitting the long prompt, and fail closed if IDs do not match exactly.
Only count a retrieval as a
pass on finish_reason stop with every needle present in final content.

transformers is imported lazily inside load_tokenizer so the offline project
venv can run the unit tests without the heavy stack; the real run uses
local_files_only=True and never writes into the model tree.
"""
import argparse
import hashlib
import json
import re
import urllib.request
from collections.abc import Mapping
from pathlib import Path

UNIT = 'lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor. '
CODE_RE = re.compile(r'^SKU-[A-Za-z0-9-]+$')
CASES = ('single25', 'single50', 'single90', 'multi33_66', 'single_near', 'multi_near')
DEPTH = {'single25': 0.25, 'single50': 0.50, 'single90': 0.90,
         'multi33_66': 0.90, 'single_near': 1.0, 'multi_near': 1.0}


def make(n: int, tag: str, multi: bool):
    hay = UNIT * n
    if multi:
        codes = [f'SKU-ALPHA-{tag}', f'SKU-BRAVO-{tag}']
        edits = [(int(len(hay) / 3), f' The ALPHA crate SKU is {codes[0]}. '),
                 (int(len(hay) * 2 / 3), f' The BRAVO crate SKU is {codes[1]}. ')]
        for pos, text in reversed(edits):
            hay = hay[:pos] + text + hay[pos:]
        question = '\nList the ALPHA and BRAVO inventory SKUs in that order.'
    else:
        codes = [f'SKU-{tag}']
        pos = int(len(hay) * 0.9)
        hay = hay[:pos] + f' The blue crate SKU is {codes[0]}. ' + hay[pos:]
        question = '\nWhat is the blue crate inventory SKU? Reply with the SKU only.'
    return f'Inventory case {tag}.\n' + hay + question, codes


def template_ids(tokenizer, prompt: str):
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                        tokenize=True, add_generation_prompt=True,
                                        reasoning_effort='low')
    # transformers >=5 returns a BatchEncoding (input_ids + attention_mask);
    # normalize to the flat id sequence of the single-message row.
    if isinstance(ids, Mapping):
        ids = ids['input_ids']
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)


def count(tokenizer, prompt: str, bos_overhead: int = 0) -> int:
    if bos_overhead not in (0, 1):
        raise ValueError('unproven BOS overhead')
    return len(template_ids(tokenizer, prompt)) + bos_overhead


def prove_server_bos(tokenizer, base: str, proof_path: Path) -> int:
    """Gate the exact one-token server prefix with independent raw text IDs."""
    probe = 'What is the blue crate SKU? Reply SKU only.'
    messages = [{'role': 'user', 'content': probe}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              reasoning_effort='low')
    if not isinstance(rendered, str) or not rendered:
        raise ValueError('BOS proof requires rendered chat text')
    local = template_ids(tokenizer, probe)
    if not local or local[0] == 1:
        raise ValueError('BOS proof requires template IDs without a leading BOS')
    req = urllib.request.Request(base + '/v1/token/encode',
                                 data=json.dumps({'text': rendered, 'add_bos_token': True,
                                                  'encode_special_tokens': True}).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=20) as f:
        server = json.load(f)
    actual = server.get('tokens')
    if (server.get('length') != len(local) + 1 or
            not isinstance(actual, list) or actual != [1] + local):
        raise ValueError('BOS proof failed: server IDs differ from one BOS plus local template')
    proof = {'status': 'PASS', 'render_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
             'local_count': len(local), 'server_count': len(actual), 'inserted_bos_id': 1,
             'local_first': local[:8], 'server_first': actual[:9],
             'local_last': local[-8:], 'server_last': actual[-8:]}
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    return 1


def fit(tokenizer, target: int, tag: str, multi: bool, bos_overhead: int = 0):
    c100 = count(tokenizer, make(100, tag, multi)[0], bos_overhead)
    c200 = count(tokenizer, make(200, tag, multi)[0], bos_overhead)
    slope = max(1.0, (c200 - c100) / 100)
    n = max(1, int((target - c100 + 100 * slope) / slope))
    actual = c100
    for _ in range(8):
        prompt, codes = make(n, tag, multi)
        actual = count(tokenizer, prompt, bos_overhead)
        if target - 100 <= actual <= target:
            return prompt, codes, actual
        n = max(1, n + int((target - actual) / slope) - (actual > target))
    raise ValueError(f'token fit missed target {target}: {actual}')


def audit(actual: int, data: dict, codes: list[str]):
    usage = data.get('usage') or {}
    if usage.get('prompt_tokens') != actual:
        raise ValueError(f'token parity failed: local {actual} vs api {usage.get("prompt_tokens")}')
    choices = data.get('choices') or []
    if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
        raise ValueError('incomplete generation')
    text = (choices[0].get('message') or {}).get('content') or ''
    if not all(code in text for code in codes):
        raise ValueError(f'needle retrieval failed; wanted {codes} in {text!r}')
    return text


def payload_for(model, prompt, reserve):
    return {'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0,
            'max_tokens': reserve,
            'reasoning_effort': 'low',
            'chat_template_kwargs': {'reasoning_effort': 'low'}}


def ask(base, model, prompt, reserve):
    req = urllib.request.Request(base + '/v1/chat/completions',
                                 data=json.dumps(payload_for(model, prompt, reserve)).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3600) as f:
        return json.load(f)


def load_tokenizer(target):
    from transformers import AutoTokenizer  # heavy import kept off the test path
    return AutoTokenizer.from_pretrained(target, local_files_only=True, trust_remote_code=True)


def job_plan(window, reserve, only):
    if only != 'all' and only not in CASES:
        raise ValueError(f'unknown case: {only}')
    if only in ('single_near', 'multi_near'):
        target = window - reserve - 96
        if target <= 0:
            raise ValueError('no near-window budget after output reserve')
        return [(only, 1.0, only == 'multi_near', target)]
    return [(label, DEPTH[label], multi, int((window - reserve) * DEPTH[label]))
            for label, multi in (('single25', False), ('single50', False),
                                 ('single90', False), ('multi33_66', True))
            if only in ('all', label)]


def check_window(actual, reserve, window):
    if actual + reserve > window:
        raise ValueError(f'request exceeds served window: {actual}+{reserve} > {window}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--base', default='http://127.0.0.1:5013')
    parser.add_argument('--model', default='target')
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--only', choices=('all',) + CASES, default='all')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    reserve = 2048
    tokenizer = load_tokenizer(args.target)
    proof_path = args.out.with_name(args.out.stem + '.bos-proof.json')
    bos_overhead = prove_server_bos(tokenizer, args.base, proof_path)
    rows = []
    for label, _fraction, multi, target in job_plan(args.window, reserve, args.only):
        prompt, codes, actual = fit(tokenizer, target, label, multi, bos_overhead)
        check_window(actual, reserve, args.window)
        data = ask(args.base, args.model, prompt, reserve)
        text = audit(actual, data, codes)
        rows.append({'case': label,
                     'prompt_tokens': actual,
                     'api_prompt_tokens': (data.get('usage') or {}).get('prompt_tokens'),
                     'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                     'completion_tokens': (data.get('usage') or {}).get('completion_tokens'),
                     'finish': 'stop',
                     'codes': codes,
                     'clean_completion': '<|assistant|>' not in text,
                     'bos_proof': str(proof_path),
                     'answer': text[:300]})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'window': args.window, 'rows': rows}, indent=2) + '\n')
    print(f'NIAH_EXACT_PASS window={args.window} rows={len(rows)}')


if __name__ == '__main__':
    main()
