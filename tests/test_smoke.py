import base64
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from scripts import smoke

STOP_42 = {'choices': [{'finish_reason': 'stop', 'message': {'content': '42'}}]}


class FinalTest(unittest.TestCase):
    def test_empty_or_truncated_is_not_success(self):
        with self.assertRaises(ValueError):
            smoke.final({'choices': [{'finish_reason': 'length', 'message': {'content': ''}}]})
        with self.assertRaises(ValueError):
            smoke.final({'choices': [{'finish_reason': 'length', 'message': {'content': 'partial answer'}}]})
        with self.assertRaises(ValueError):
            smoke.final({'choices': [{'finish_reason': 'stop', 'message': {'content': '   '}}]})
        with self.assertRaises(ValueError):
            smoke.final({'choices': [{'finish_reason': 'stop', 'message': {}}]})
        with self.assertRaises(ValueError):
            smoke.final({'choices': []})
        self.assertEqual(smoke.final(STOP_42), '42')
        self.assertEqual(
            smoke.final({'choices': [{'finish_reason': 'stop', 'message': {'content': ' 42\n'}}]}), '42')


class TextGateTest(unittest.TestCase):
    def test_requires_42_in_stopped_final_content(self):
        self.assertEqual(
            smoke.text_gate({'choices': [{'finish_reason': 'stop', 'message': {'content': 'The answer is 42.'}}]}),
            'The answer is 42.')
        with self.assertRaisesRegex(ValueError, 'text gate'):
            smoke.text_gate({'choices': [{'finish_reason': 'stop', 'message': {'content': 'forty-two in words only'}}]})
        with self.assertRaises(ValueError):
            smoke.text_gate({'choices': [{'finish_reason': 'length', 'message': {'content': '42 so far and th'}}]})


class ToolGateTest(unittest.TestCase):
    def test_requires_real_get_time_tool_call_not_http_200(self):
        ok = {'choices': [{'finish_reason': 'tool_calls', 'message': {
            'content': '',
            'tool_calls': [{'id': '1', 'type': 'function',
                            'function': {'name': 'get_time', 'arguments': '{}'}}]}}]}
        self.assertEqual(smoke.tool_gate(ok)['name'], 'get_time')
        http_200_no_call = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': 'It is 17:00.', 'tool_calls': []}}]}
        with self.assertRaisesRegex(ValueError, 'tool_call'):
            smoke.tool_gate(http_200_no_call)
        wrong_name = json.loads(json.dumps(ok))
        wrong_name['choices'][0]['message']['tool_calls'][0]['function']['name'] = 'other_tool'
        with self.assertRaisesRegex(ValueError, 'tool_call'):
            smoke.tool_gate(wrong_name)
        with self.assertRaises(ValueError):
            smoke.tool_gate({'choices': [{'finish_reason': 'length', 'message': {'content': 'x'}}]})

    def test_tool_definition_uses_required_choice(self):
        self.assertEqual(smoke.TOOL['type'], 'function')
        self.assertEqual(smoke.TOOL['function']['name'], 'get_time')


class VisionGateTest(unittest.TestCase):
    def test_requires_word_and_number_in_final_content(self):
        ok = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'I can see BANANA and the number 42.'}}]}
        self.assertIn('BANANA', smoke.vision_gate(ok))
        with self.assertRaisesRegex(ValueError, 'vision gate'):
            smoke.vision_gate({'choices': [{'finish_reason': 'stop',
                                            'message': {'content': 'the word is there but no digits'}}]})
        with self.assertRaisesRegex(ValueError, 'vision gate'):
            smoke.vision_gate({'choices': [{'finish_reason': 'stop', 'message': {'content': 'only 42 here'}}]})
        with self.assertRaises(ValueError):
            smoke.vision_gate({'choices': [{'finish_reason': 'length', 'message': {'content': 'BANANA 4'}}]})


class PngTest(unittest.TestCase):
    def test_in_memory_png_data_uri_when_pillow_available(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest('Pillow unavailable in offline project venv; real path runs in-container')
        uri = smoke.vision_png()
        self.assertTrue(uri.startswith('data:image/png;base64,'))
        raw = base64.b64decode(uri.split(',', 1)[1])
        self.assertEqual(raw[:8], b'\x89PNG\r\n\x1a\n')
        from PIL import Image
        image = Image.open(io.BytesIO(raw))
        self.assertEqual(image.size, (384, 256))


class PayloadContractTest(unittest.TestCase):
    def test_request_posts_required_fields_to_container_url(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured['url'] = req.full_url
            captured['data'] = json.loads(req.data.decode())
            captured['timeout'] = timeout
            return io.BytesIO(json.dumps({'choices': [], 'usage': {}}).encode())

        with mock.patch.object(smoke.urllib.request, 'urlopen', fake_urlopen):
            smoke.request([{'role': 'user', 'content': 'hi'}], [smoke.TOOL])
        self.assertEqual(captured['url'], smoke.BASE_URL + '/v1/chat/completions')
        payload = captured['data']
        self.assertEqual(payload['model'], 'target')
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['max_tokens'], 8192)
        self.assertEqual(payload['chat_template_kwargs'], {'reasoning_effort': 'low'})
        self.assertEqual(payload['tool_choice'], 'required')
        self.assertEqual(payload['tools'], [smoke.TOOL])


class ModelsCheckTest(unittest.TestCase):
    def test_target_must_be_listed(self):
        with mock.patch.object(smoke.urllib.request, 'urlopen',
                               return_value=io.BytesIO(json.dumps({'data': [{'id': 'target'}]}).encode())):
            smoke.models_ok()
        with mock.patch.object(smoke.urllib.request, 'urlopen',
                               return_value=io.BytesIO(json.dumps({'data': [{'id': 'other'}]}).encode())):
            with self.assertRaisesRegex(ValueError, 'target'):
                smoke.models_ok()


class MainFlowTest(unittest.TestCase):
    def test_end_to_end_with_mocked_http_and_placeholder_png(self):
        tool_resp = {'choices': [{'finish_reason': 'tool_calls', 'message': {
            'content': '', 'tool_calls': [{'function': {'name': 'get_time', 'arguments': '{}'}}]}}]}
        vision_resp = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'It reads BANANA 42.'}}]}
        calls = []

        def fake_request(messages, tools=None):
            calls.append((messages, tools))
            if tools is not None:
                return tool_resp
            if isinstance(messages[0]['content'], list):
                return vision_resp
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': '42'}}]}

        printed = io.StringIO()
        with mock.patch.object(smoke, 'models_ok', return_value=None), \
                mock.patch.object(smoke, 'request', side_effect=fake_request), \
                mock.patch.object(smoke, 'vision_png', return_value='data:image/png;base64,AAAA'), \
                redirect_stdout(printed):
            smoke.main()
        out = printed.getvalue()
        self.assertIn('SMOKE_OK', out)
        self.assertIn('text=42', out)
        self.assertIn('tool=get_time', out)
        self.assertIn('vision=BANANA_42', out)
        vision_calls = [m for m, t in calls if t is None and isinstance(m[0]['content'], list)]
        self.assertEqual(len(vision_calls), 1)
        self.assertEqual(vision_calls[0][0]['content'][1]['image_url']['url'], 'data:image/png;base64,AAAA')
        tool_calls = [(m, t) for m, t in calls if t is not None]
        self.assertEqual(len(tool_calls), 1)
