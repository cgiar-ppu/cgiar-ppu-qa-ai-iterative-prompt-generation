# task_generator.py

import itertools
from tiktoken import encoding_for_model
import pandas as pd
import csv

def estimate_token_count(prompt_text, input_text, model_name):
    encoding = encoding_for_model(model_name)
    total_text = prompt_text.replace('[INPUT_TEXT]', input_text)
    tokens = encoding.encode(total_text)
    return len(tokens)

# task_generator.py

def generate_task_list(df_input, prompts, models, max_token_limit=80000):
    tasks = []
    for _, row in df_input.iterrows():
        input_text = row['input_text']
        result_code = row['Result code']
        for prompt_id, prompt in prompts.items():
            prompt_text = prompt['text']
            perspective = prompt.get('perspective', '')
            impact_area = prompt.get('impact_area', '')
            for model_name in models:
                # Estimate token count
                token_count = estimate_token_count(prompt['text'], row['input_text'], model_name)
                if token_count > max_token_limit:
                    print(f"Skipping task due to token limit: {prompt_id} on {model_name} for result code {result_code}")
                    continue
                task = {
                    'result_code': result_code,
                    'prompt_id': prompt_id,
                    'prompt_text': prompt_text,
                    'model_name': model_name,
                    'perspective': perspective,
                    'impact_area': impact_area,
                    'input_text': input_text,
                    'token_count': token_count
                }
                tasks.append(task)
    return tasks


def save_task_list(tasks, task_list_excel, task_list_csv):
    df_tasks = pd.DataFrame(tasks)
    df_tasks.to_excel(task_list_excel, index=False)
    df_tasks.to_csv(task_list_csv, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')