from openai import OpenAI

class LLM_node:

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required":{
                "system_prompt" : ("STRING",{"default":"","multiline": True,},),
                "user_query": ("STRING", {"default":"","multiline": True,},),
                "model_name": (["gpt-3.5-turbo","gpt-4o","gpt-4-turbo","gpt-4",""],),
                "api_key": ("STRING",{"default":""},)
            },
            "optional":{
                "max_tokens" : ("INT", {"default":250,"min":10,"max":2000,"step":10},),
                "temperature" : ("FLOAT", {"default":0,"min":0,"max":1,"step":0.1}),
                "timeout": ("INT", {"default":10,"min":1,"max":200,"step":1},),
            }
        }

    CATEGORY = "DeepFuze"
    RETURN_TYPES = ("NEW_STRING",)
    RETURN_NAMES = ("LLM_RESPONSE",)
    FUNCTION = "run_llm"

    def run_llm(self, system_prompt, user_query, model_name,temperature,api_key,max_tokens,timeout):
        client = OpenAI(api_key=api_key)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                    ],
                    # stop = stop.split(","),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                ).to_dict()
            print(response)
            result = response["choices"][0]["message"]["content"]
        except Exception as e:
            raise ValueError(e)
        return (result,)
