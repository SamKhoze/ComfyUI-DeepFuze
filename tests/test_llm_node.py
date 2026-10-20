"""Offline LLM-node tests; run with python tests/test_llm_node.py."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI


# Load only the text node, without importing ComfyUI or the media/ML dependencies.
spec = importlib.util.spec_from_file_location('llm_node', Path(__file__).resolve().parents[1] / 'llm_node.py')
llm_node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_node)


class LLMNodeTests(unittest.TestCase):
	def test_picker_exposes_current_minimax_models_in_order(self):
		models = llm_node.LLM_node.INPUT_TYPES()['required']['model_name'][0]
		self.assertEqual([model for model in models if model.startswith('MiniMax-')],
			['MiniMax-M3.1-Flash', 'MiniMax-M3'])
		self.assertEqual([model for model in models if not model.startswith('MiniMax-')],
			['gpt-3.5-turbo', 'gpt-4o', 'gpt-4-turbo', 'gpt-4', ''])

	def test_saved_and_custom_minimax_selections_remain_valid(self):
		for model in ('MiniMax-M3.1-Flash', 'MiniMax-M3', 'MiniMax-M2.7', 'MiniMax-M2.5', 'MiniMax-custom'):
			with self.subTest(model=model):
				self.assertIs(llm_node.LLM_node.VALIDATE_INPUTS(model), True)
		self.assertIs(llm_node.LLM_node.VALIDATE_INPUTS('gpt-4o'), True)
		self.assertEqual(llm_node.LLM_node.VALIDATE_INPUTS('unknown'), 'Unknown model_name')
		self.assertEqual(llm_node.LLM_node.VALIDATE_INPUTS(None), 'Unknown model_name')

	def assert_request(self, model, region, expected_url, max_tokens=250, user_query='Write a dialogue.'):
		requests = []

		def respond(request):
			requests.append(request)
			return httpx.Response(200, json={
				'id': 'offline-response', 'object': 'chat.completion', 'created': 0,
				'model': model,
				'choices': [{'index': 0, 'finish_reason': 'stop',
					'message': {'role': 'assistant', 'content': 'A dialogue.'}}],
			})

		with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
			def client_factory(**kwargs):
				return OpenAI(http_client=transport, max_retries=0, **kwargs)

			options = {} if region is None else {'minimax_region': region}
			with patch.object(llm_node, 'OpenAI', side_effect=client_factory), patch('builtins.print'):
				result = llm_node.LLM_node().run_llm(
					'You write dialogue.', user_query, model, 0, 'offline-test-key', max_tokens, 10, **options)

		self.assertEqual(result, ('A dialogue.',))
		self.assertEqual(len(requests), 1)
		self.assertEqual(requests[0].method, 'POST')
		self.assertEqual(str(requests[0].url), expected_url)
		# Exact payload also guards against disabled thinking, unsupported effort,
		# service_tier, or reasoning_split accidentally reaching the API.
		self.assertEqual(json.loads(requests[0].content), {
			'model': model,
			'messages': [
				{'role': 'system', 'content': 'You write dialogue.'},
				{'role': 'user', 'content': user_query},
			],
			'temperature': 0, 'max_tokens': max_tokens,
		})

	def test_minimax_requests_preserve_model_and_regional_endpoint(self):
		for model in ('MiniMax-M3.1-Flash', 'MiniMax-M3', 'MiniMax-M2.7', 'MiniMax-custom'):
			for region, host in (('global_en', 'api.minimax.io'), ('cn_zh', 'api.minimaxi.com')):
				with self.subTest(model=model, region=region):
					self.assert_request(model, region, f'https://{host}/v1/chat/completions')

	def test_default_region_and_explicit_token_budget(self):
		self.assert_request(llm_node.MINIMAX_MODEL_IDS[0], None,
			'https://api.minimax.io/v1/chat/completions', max_tokens=16000)

	def test_other_models_keep_existing_endpoint_and_defaults(self):
		model = llm_node.LLM_node.INPUT_TYPES()['required']['model_name'][0][0]
		self.assert_request(model, 'cn_zh', 'https://api.openai.com/v1/chat/completions')

	def test_content_parts_pass_through_unchanged(self):
		content = [
			{'type': 'text', 'text': 'Describe this scene.'},
			{'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png'}},
			{'type': 'video_url', 'video_url': {'url': 'https://example.com/video.mp4'}},
		]
		self.assert_request('MiniMax-M3.1-Flash', None,
			'https://api.minimax.io/v1/chat/completions', user_query=content)


if __name__ == '__main__':
	unittest.main()
