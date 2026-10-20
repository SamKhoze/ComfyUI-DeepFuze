from openai import OpenAI

# MiniMax is OpenAI-compatible. The OpenAI-compatible base URLs and model ids
# come from the MiniMax provider configuration so the LLM node can target the
# global and CN regional endpoints instead of the default OpenAI endpoint.
MINIMAX_OPENAI_BASE_URL_GLOBAL = 'https://api.minimax.io/v1'
MINIMAX_OPENAI_BASE_URL_CN = 'https://api.minimaxi.com/v1'
MINIMAX_MODEL_IDS = ['MiniMax-M3.1-Flash', 'MiniMax-M3']
MINIMAX_REGIONS = ['global_en', 'cn_zh']


class LLM_node:

	@classmethod
	def INPUT_TYPES(self):
		return {
			'required': {
				'system_prompt': ('STRING', {'default': '', 'multiline': True}),
				'user_query': ('STRING', {'default': '', 'multiline': True}),
				'model_name': (['gpt-3.5-turbo', 'gpt-4o', 'gpt-4-turbo', 'gpt-4', ''] + MINIMAX_MODEL_IDS,),
				'api_key': ('STRING', {'default': ''}),
			},
			'optional': {
				'max_tokens': ('INT', {'default': 250, 'min': 10, 'max': 2000, 'step': 10}),
				'temperature': ('FLOAT', {'default': 0, 'min': 0, 'max': 1, 'step': 0.1}),
				'timeout': ('INT', {'default': 10, 'min': 1, 'max': 200, 'step': 1}),
				'minimax_region': (MINIMAX_REGIONS, {'default': 'global_en'}),
			},
		}

	CATEGORY = 'DeepFuze'
	RETURN_TYPES = ('NEW_STRING',)
	RETURN_NAMES = ('LLM_RESPONSE',)
	FUNCTION = 'run_llm'

	@classmethod
	def VALIDATE_INPUTS(cls, model_name):
		# Saved workflows may explicitly select a MiniMax model no longer in the picker.
		if isinstance(model_name, str) and model_name.startswith('MiniMax-'):
			return True
		if model_name in cls.INPUT_TYPES()['required']['model_name'][0]:
			return True
		return 'Unknown model_name'

	@staticmethod
	def get_base_url(model_name: str, minimax_region: str):
		# MiniMax models are routed through the OpenAI-compatible endpoint of
		# the selected regional base URL; other models use the default OpenAI
		# endpoint.
		if isinstance(model_name, str) and model_name.startswith('MiniMax-'):
			return MINIMAX_OPENAI_BASE_URL_CN if minimax_region == 'cn_zh' else MINIMAX_OPENAI_BASE_URL_GLOBAL
		return None

	def run_llm(self, system_prompt, user_query, model_name, temperature, api_key, max_tokens, timeout, minimax_region='global_en'):
		base_url = self.get_base_url(model_name, minimax_region)
		if base_url:
			client = OpenAI(api_key=api_key, base_url=base_url)
		else:
			client = OpenAI(api_key=api_key)
		try:
			response = client.chat.completions.create(
				model=model_name,
				messages=[
					{'role': 'system', 'content': system_prompt},
					{'role': 'user', 'content': user_query}
				],
				# stop = stop.split(","),
				temperature=temperature,
				max_tokens=max_tokens,
				timeout=timeout
			).to_dict()
			print(response)
			result = response['choices'][0]['message']['content']
		except Exception as e:
			raise ValueError(e)
		return (result,)
