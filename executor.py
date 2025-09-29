# executor.py

from openai import OpenAI
import threading
import pandas as pd
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
#api_key = os.getenv('OPENAI_API_KEY')

# List of models that require the simplified API call
simplified_models = ['o1-preview', 'o3']

def get_client(model_name):
    """
    Create and return an OpenAI client based on the model name.
    Handles different API keys and base URLs.
    """
    if model_name.startswith('grok-'):
        api_key = os.getenv('XAI_API_KEY')
        base_url = "https://api.x.ai/v1"
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = None  # Use default OpenAI base URL
    
    return OpenAI(api_key=api_key, base_url=base_url)

def execute_task(task):
    """
    Execute a single task by sending a request to the OpenAI API.
    """
    client = get_client(task['model_name'])

    # Replace placeholders in the prompt text
    prompt_text = task['prompt_text'].replace('[INPUT_TEXT]', task['input_text'])
    
    # Prepare messages
    if task['model_name'] in simplified_models:
        # For simplified models, only include the user message
        messages = [
            {"role": "user", "content": prompt_text}
        ]
    else:
        # For other models, include the system role and other parameters
        role = "You are an assistant that will closely follow the instruction provided next and respond in a concise way by providing a direct answer without too many details."
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": prompt_text}
        ]
    
    try:
        if task['model_name'] in simplified_models:
            # Use simplified API call for specific models
            api_params = {
                "model": task['model_name'],
                "messages": messages
            }
            
            # Add reasoning_effort parameter only for o3-mini model
            if task['model_name'] == 'o3-mini':
                api_params["reasoning_effort"] = "high"
                
            response = client.chat.completions.create(**api_params)
        else:
            # Use the standard API call with additional parameters
            api_params = {
                "model": task['model_name'],
                "messages": messages,
                "temperature": 0,
                "max_tokens": 10000,  # Increased to 10000
                "response_format": {"type": "text"}
            }
            if task['model_name'] != 'grok-4':
                api_params["presence_penalty"] = 0
                api_params["frequency_penalty"] = 0
            if task['model_name'].startswith('grok-'):
                api_params["top_p"] = 0.1  # Positive value required for xAI API
            else:
                api_params["top_p"] = 0
            response = client.chat.completions.create(**api_params)
        output = response.choices[0].message.content.strip()
        result = {
            'result_code': task['result_code'],
            'prompt_id': task['prompt_id'],
            'model_name': task['model_name'],
            'impact_area': task['impact_area'],
            'model_output': output,
            'token_count': task['token_count'],
            'timestamp': pd.Timestamp.now()
        }
        return result
    except Exception as e:
        print(f"Error querying API for task {task['result_code']}, {task['prompt_id']}, {task['model_name']}: {e}")
        return None

def execute_tasks_concurrently(tasks, max_workers=8, progress_callback=None):
    """
    Execute tasks concurrently using ThreadPoolExecutor, updating progress after each task.
    """
    results = []
    total_tasks = len(tasks)
    completed_tasks = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(execute_task, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Task {task} generated an exception: {e}")
            finally:
                completed_tasks += 1
                if progress_callback:
                    progress_callback(completed_tasks)
    
    return results