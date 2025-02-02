# main.py

import os
import argparse
from data_loader import load_data, load_existing_results
from task_generator import generate_task_list, save_task_list
from executor import execute_tasks_concurrently
from evaluator import evaluate_results
from prompt_improver import improve_prompts
from logger import ResultLogger
import config
import prompt_techniques
import pandas as pd
from datetime import datetime
from response_extractor import add_score_explanation_columns
import time

start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    parser = argparse.ArgumentParser(description='Run AI evaluation tasks.')
    parser.add_argument('--prompts', nargs='*', help='List of prompt IDs to include. If omitted, all active prompts will be used.')
    parser.add_argument('--models', nargs='*', help='List of model names to include. If omitted, all models in config.MODELS will be used.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--list', action='store_true', help='List all available prompts and models.')
    args = parser.parse_args()

    # Load configurations
    prompts = config.PROMPTS
    models = config.MODELS

    # Filter prompts based on command-line arguments
    if args.prompts:
        prompts = {pid: p for pid, p in prompts.items() if pid in args.prompts}
    else:
        # Use only active prompts if no arguments are provided
        prompts = {pid: p for pid, p in prompts.items() if p.get('active', True)}

    # Filter models based on command-line arguments
    if args.models:
        models = [m for m in models if m in args.models]

    if args.verbose:
        print("Using prompts:")
        for pid in prompts:
            print(f" - {pid}")
        print("Using models:")
        for model in models:
            print(f" - {model}")

    if args.list:
        print("Available Prompts:")
        for pid in config.PROMPTS.keys():
            print(f" - {pid}")
        print("\nAvailable Models:")
        for model in config.MODELS:
            print(f" - {model}")
        return
    
    # Stage 1: Task List Creation
    input_csv = 'input//Joined_Processed_Evidence_PRMS_ExpertsScore.csv'
    task_list_csv = f'output//task_list_{timestamp}.csv'
    task_list_excel = f'output//task_list_{timestamp}.xlsx'
    output_csv = f'output//results_{timestamp}.csv'
    metrics_csv = f'output//metrics_{timestamp}.csv'

    df_input = load_data(input_csv)
    # Create a mapping from 'Result code' to 'Impact Area checked'
    result_code_to_impact_areas = df_input.groupby('Result code')['Impact Area checked'].apply(list).to_dict()
    # Deduplicate input data based on 'Result code' and relevant text fields
    df_unique_input = df_input.drop_duplicates(subset=['Result code', 'Title', 'Description', 'Evidence Abstract Text', 'Evidence Parsed Text'])

    existing_results = load_existing_results(output_csv)
    tasks = generate_task_list(df_unique_input, prompts, models)

    # Filter out completed tasks for resumability
    tasks_to_run = [task for task in tasks if (
        task['result_code'], task['prompt_id'], task['model_name']
    ) not in existing_results]

    # Save task list
    save_task_list(tasks_to_run, task_list_excel, task_list_csv)

    # Stage 2: Execution of Tasks
    logger = ResultLogger(output_csv)
    results = execute_tasks_concurrently(tasks_to_run)

    # Log results
    for result in results:
        logger.log_result(result)

    # Response_extractor
    results_df = pd.read_csv(output_csv)
    results_df = add_score_explanation_columns(results_df, 'model_output')
    results_df.to_csv(output_csv, index=False)

    # Stage 3: Evaluation and Metrics Calculation
    evaluate_results(output_csv, input_csv, metrics_csv)

    # Stage 4: Automated Prompt Improvement
    improved_prompts = improve_prompts(config.PROMPTS, prompt_techniques.PROMPT_TECHNIQUES)

    # Update prompts with improved ones
    config.PROMPTS.update(improved_prompts)

    # Repeat stages as needed...

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()