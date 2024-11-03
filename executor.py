# executor.py

from openai import OpenAI
import threading
import pandas as pd
import json
import time

client = OpenAI(api_key="sk-OwIfHlbOIes9PqMF-mip_JTSzJ4qkYx-g8YjjoIKXXT3BlbkFJyd1dpfID6zsTfjMNZzE98xEZEdFU_ppS_LvceWDPcA")

# List of models that require the simplified API call
simplified_models = ['o1-preview', 'o1-mini']

def execute_task(task):
    """
    Execute a single task by sending a request to the OpenAI API.
    """
    # Replace placeholders in the prompt text
    prompt_text = task['prompt_text'].replace('[INPUT_TEXT]', task['input_text'])
    prompt_text = prompt_text.replace('[PERSPECTIVE]', task['perspective'])
    
    # Prepare messages
    if task['model_name'] in simplified_models:
        # For simplified models, only include the user message
        messages = [
            {"role": "user", "content": prompt_text}
        ]
    else:
        # For other models, include the system role and other parameters
        role = "You are an assistant that will closely follow the instruction provided next and respond in a concise way by providing a direct answer and also a very brief explanation without too many details."
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": prompt_text}
        ]
    
    try:
        if task['model_name'] in simplified_models:
            # Use simplified API call for specific models
            response = openai.ChatCompletion.create(
                model=task['model_name'],
                messages=messages
            )
        else:
            # Use the standard API call with additional parameters
            response = openai.ChatCompletion.create(
                model=task['model_name'],
                messages=messages,
                temperature=0,  # Adjust as needed
                max_tokens=500,
                top_p=0,
                frequency_penalty=0,
                presence_penalty=0
            )
        output = response.choices[0].message['content'].strip()
        result = {
            'result_code': task['result_code'],
            'prompt_id': task['prompt_id'],
            'model_name': task['model_name'],
            'perspective': task['perspective'],
            'impact_area': task['impact_area'],
            'model_output': output,
            'token_count': task['token_count'],
            'timestamp': pd.Timestamp.now()
        }
        return result
    except Exception as e:
        print(f"Error querying OpenAI for task {task['result_code']}, {task['prompt_id']}, {task['model_name']}: {e}")
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