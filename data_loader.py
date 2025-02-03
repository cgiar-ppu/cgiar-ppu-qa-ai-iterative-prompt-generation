# data_loader.py

import pandas as pd
import streamlit as st
import os

def load_data(input_file, combine_evidence=False):
    """
    Load data from a CSV file path or a file-like object and preprocess it.
    """
    # Check if input_file is a file-like object or a string path
    if isinstance(input_file, str):
        # If it's a string, assume it's a file path
        file_extension = input_file.split('.')[-1].lower()
    else:
        # If it's a file-like object, access the .name attribute
        file_extension = input_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            df = pd.read_csv(input_file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(input_file)
        else:
            # Attempt to read as CSV first, then Excel if it fails
            try:
                df = pd.read_csv(input_file)
            except Exception:
                if not isinstance(input_file, str):
                    input_file.seek(0)  # Reset file pointer if it's a file-like object
                df = pd.read_excel(input_file)
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
        st.stop()

    if combine_evidence:
        try:
            df = combine_rows_by_result_code(df)  # call the new function
        except ValueError as e:
            st.error(str(e))
            st.stop()

    # Check for required columns in the DataFrame
    possible_columns = ['Title', 'Description', 'Evidence Abstract Text', 'Evidence Parsed Text']
    available_columns = [col for col in possible_columns if col in df.columns]

    if not available_columns:
        st.error("None of the expected text columns ('Title', 'Description', 'Evidence Abstract Text', 'Evidence Parsed Text') are present in the uploaded data. At least one is required to create 'input_text'.")
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

def combine_rows_by_result_code(df):
    """
    Group rows by 'Result code' and concatenate all 'Evidence Extracted Text'
    for each code into one row. Keeps one row per unique 'Result code'.

    If your spreadsheet also needs other columns aggregated,
    you can modify the .agg(...) dictionary accordingly.
    """
    # Safety check
    required_cols = ['Result code', 'Evidence Parsed Text']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"To combine rows by result code, the dataframe must have '{col}' column."
            )

    # Example aggregator:
    # 1) Keep only the first non-null of certain columns (like Title, Description, etc.).
    # 2) Concatenate the text in 'Evidence Extracted Text' separated by a period and new paragraphs.
    agg_dict = {
        'Title': 'first',  # or you can choose to combine them differently if you want
        'Description': 'first',
        'Evidence Abstract Text': 'first',
        'Evidence Parsed Text': lambda series: ".\n\n".join(
            str(x) for x in series.dropna()
        )
    }

    # Before grouping, ensure these columns exist. If not, remove them from aggregator:
    actual_agg_dict = {}
    for col, how in agg_dict.items():
        if col in df.columns:
            actual_agg_dict[col] = how

    # Do the grouping and aggregation
    df_combined = (
        df
        .groupby('Result code', as_index=False)
        .agg(actual_agg_dict)
    )

    return df_combined
