# prompt_improver.py

from copy import deepcopy

def improve_prompts(prompts, prompt_techniques):
    new_prompts = {}
    for prompt_id, prompt in prompts.items():
        for technique in prompt_techniques:
            new_prompt = deepcopy(prompt)
            new_prompt['id'] = f"{prompt_id}_{technique['name'].replace(' ', '_')}"
            new_prompt['text'] = technique['function'](prompt['text'])
            new_prompts[new_prompt['id']] = new_prompt
    return new_prompts