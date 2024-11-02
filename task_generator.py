# task_generator.py

import itertools
from tiktoken import encoding_for_model
import pandas as pd

def estimate_token_count(prompt_text, input_text, model_name):
    encoding = encoding_for_model(model_name)
    total_text = prompt_text.replace('[INPUT_TEXT]', input_text)
    tokens = encoding.encode(total_text)
    return len(tokens)

def generate_task_list(df, prompts, models, max_token_limit=8000):
    tasks = []
    for _, row in df.iterrows():
        for prompt_key, prompt_value in prompts.items():
            for model_name in models:
                # Estimate token count
                token_count = estimate_token_count(prompt_value['text'], row['input_text'], model_name)
                if token_count > max_token_limit:
                    print(f"Skipping task due to token limit: {row['Result code']}, {prompt_key}, {model_name}")
                    continue  # Skip tasks exceeding token limits
                task = {
                    'result_code': row['Result code'],
                    'prompt_id': prompt_value['id'],
                    'prompt_text': prompt_value['text'],
                    'model_name': model_name,
                    'perspective': prompt_value['perspective'],
                    'input_text': row['input_text'],
                    'token_count': token_count
                }
                tasks.append(task)
    return tasks

def save_task_list(tasks, task_list_csv):
    df_tasks = pd.DataFrame(tasks)
    df_tasks.to_csv(task_list_csv, index=False)