# data_loader.py

import pandas as pd
import streamlit as st

def load_data(input_csv):
    """
    Load data from the CSV file or file-like object and preprocess it.
    """
    df = pd.read_csv(input_csv)
    possible_columns = ['Title', 'Description', 'Evidence_Abstract_Text', 'Evidence_Parsed_Text']
    available_columns = [col for col in possible_columns if col in df.columns]

    if not available_columns:
        st.error("None of the expected text columns ('Title', 'Description', 'Evidence_Abstract_Text', 'Evidence_Parsed_Text') are present in the uploaded data. At least one is required to create 'input_text'.")
        st.stop()

    # Concatenate available text fields
    df['input_text'] = df[available_columns].fillna('').agg(' '.join, axis=1)
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
