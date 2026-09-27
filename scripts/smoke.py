"""Semantic smoke client for the owned TabbyAPI container (Task 7).

Text, tool and vision gates all require *real* model behavior in the final
response: finish_reason stop plus answer content, a structured get_time
tool_call (an HTTP 200 alone is not a pass), and the synthetic image's word
and number repeated in final content.
"""
import base64
import io
import json
import urllib.request

BASE_URL = 'http://127.0.0.1:5000'
TIMEOUT = 900
MODEL = 'target'
TOOL = {'type': 'function',
        'function': {'name': 'get_time', 'description': 'Test clock',
                     'parameters': {'type': 'object', 'properties': {},
                                    'additionalProperties': False}}}


def final(response):
    choices = response.get('choices') or []
    if not choices:
        raise ValueError('no final content')
    choice = choices[0]
    content = ((choice.get('message') or {}).get('content')) or ''
    text = content.strip()
    if choice.get('finish_reason') != 'stop' or not text:
        raise ValueError('no final content')
    return text


def request(messages, tools=None):
    payload = {'model': MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': 8192,
               'reasoning_effort': 'low', 'chat_template_kwargs': {'reasoning_effort': 'low'}}
    if tools is not None:
        payload['tools'] = tools
        payload['tool_choice'] = 'required'
    req = urllib.request.Request(BASE_URL + '/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def models_ok():
    with urllib.request.urlopen(BASE_URL + '/v1/models', timeout=10) as r:
        listed = [x['id'] for x in json.load(r)['data']]
    if MODEL not in listed:
        raise ValueError(f'{MODEL} not served; /v1/models listed {listed}')
    return listed


def text_gate(response):
    text = final(response)
    if '42' not in text:
        raise ValueError(f'text gate failed: wanted 42 in {text!r}')
    return text


def tool_gate(response):
    choice = (response.get('choices') or [])[0]
    if choice.get('finish_reason') == 'length':
        raise ValueError('no final content')
    calls = (choice.get('message') or {}).get('tool_calls') or []
    for call in calls:
        function = call.get('function') or {}
        if function.get('name') == 'get_time':
            return function
    raise ValueError(f'tool_call gate failed: no get_time among {calls!r}')


def vision_png():
    # Imported only on the real in-container smoke path; the offline project
    # venv has no Pillow. A skipped import must never fake a vision pass.
    from PIL import Image, ImageDraw  # noqa: PLC0415
    image = Image.new('RGB', (384, 256), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 60, 344, 200), outline='red', width=6)
    draw.text((120, 110), 'BANANA 42', fill='black')
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def vision_gate(response):
    text = final(response).upper()
    if 'BANANA' not in text or '42' not in text:
        raise ValueError(f'vision gate failed: wanted BANANA 42 in {text!r}')
    return text


def vision_messages(png_data_uri):
    return [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Read the word and number in this image.'},
        {'type': 'image_url', 'image_url': {'url': png_data_uri}}]}]


def main():
    models_ok()
    text_gate(request([{'role': 'user', 'content': 'What is six times seven? Number only.'}]))
    tool_gate(request([{'role': 'user', 'content': 'Call get_time now; do not answer in prose.'}], [TOOL]))
    vision_gate(request(vision_messages(vision_png())))
    print('SMOKE_OK text=42 tool=get_time vision=BANANA_42')


if __name__ == '__main__':
    main()
