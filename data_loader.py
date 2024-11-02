# data_loader.py

import pandas as pd

def load_data(input_csv):
    """
    Load data from the CSV file and preprocess it.
    """
    df = pd.read_csv(input_csv)
    # Concatenate text fields
    df['input_text'] = df[['Title', 'Description', 'Evidence_Abstract_Text', 'Evidence_Parsed_Text']].fillna('').agg(' '.join, axis=1)
    return df

def load_existing_results(output_csv):
    """
    Load existing results to support resumability.
    """
    try:
        existing_results = pd.read_csv(output_csv)
        completed_tasks = set(zip(existing_results['result_code'], existing_results['prompt_id'], existing_results['model_name']))
        return completed_tasks
    except FileNotFoundError:
        return set()