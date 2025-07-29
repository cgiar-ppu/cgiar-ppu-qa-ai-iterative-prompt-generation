# data_loader.py

import pandas as pd
import streamlit as st
import os

def load_data(input_file, combine_evidence=False, selected_columns=None, id_column=None):
    """
    Load data from a CSV file path or a file-like object and preprocess it.
    
    Args:
        input_file: File path (string) or file-like object
        combine_evidence: Boolean to combine rows by result code
        selected_columns: List of column names to use for creating input_text
        id_column: Column name to use as unique identifier (will be renamed to 'result_code')
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

    # Handle ID column selection and standardization
    if id_column:
        if id_column not in df.columns:
            st.error(f"Selected ID column '{id_column}' not found in the data.")
            st.stop()
        # Rename the selected ID column to 'result_code' for consistency
        df = df.rename(columns={id_column: 'result_code'})
    else:
        # Fall back to looking for common ID column names
        common_id_columns = ['Result code', 'result_code', 'ID', 'id', 'Code', 'code']
        found_id_column = None
        for col in common_id_columns:
            if col in df.columns:
                found_id_column = col
                break
        
        if found_id_column:
            if found_id_column != 'result_code':
                df = df.rename(columns={found_id_column: 'result_code'})
        else:
            st.error("No ID column specified and no common ID columns found. Please select an ID column.")
            st.stop()

    if combine_evidence:
        try:
            df = combine_rows_by_result_code(df, selected_columns)  # Pass selected_columns
        except ValueError as e:
            st.error(str(e))
            st.stop()

    # Handle column selection for creating input_text
    if selected_columns:
        # Use the provided selected columns
        available_columns = [col for col in selected_columns if col in df.columns]
        if not available_columns:
            st.error(f"None of the selected columns {selected_columns} are present in the data.")
            st.stop()
    else:
        # Fall back to the original hardcoded columns for backward compatibility
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

def combine_rows_by_result_code(df, selected_columns=None):
    """
    Group rows by 'result_code' and concatenate selected columns for each code into one row. Keeps one row per unique 'result_code'.
    """
    # Safety check
    if 'result_code' not in df.columns:
        raise ValueError("To combine rows by result code, the dataframe must have 'result_code' column.")
    if not selected_columns:
        raise ValueError("Selected columns must be provided for concatenation when combining rows.")
    
    # Ensure selected_columns exist in df
    available_selected_columns = [col for col in selected_columns if col in df.columns]
    if not available_selected_columns:
        raise ValueError(f"None of the selected columns {selected_columns} are present in the data for concatenation.")
    
    # Dynamically build aggregator: concatenate for selected columns, 'first' for others
    agg_dict = {}
    all_columns = df.columns.tolist()
    for col in all_columns:
        if col == 'result_code':
            continue  # Skip grouping column
        if col in available_selected_columns:
            agg_dict[col] = lambda series: ".\n\n".join(str(x) for x in series.dropna())
        else:
            agg_dict[col] = 'first'
    
    # Do the grouping and aggregation
    df_combined = (
        df
        .groupby('result_code', as_index=False)
        .agg(agg_dict)
    )
    return df_combined

def process_dataframe_with_selected_columns(df, combine_evidence=False, selected_columns=None, id_column=None):
    """
    Process an already loaded dataframe with selected columns.
    
    Args:
        df: Already loaded pandas DataFrame
        combine_evidence: Boolean to combine rows by result code
        selected_columns: List of column names to use for creating input_text
        id_column: Column name to use as unique identifier (will be renamed to 'result_code')
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Handle ID column selection and standardization
    if id_column:
        if id_column not in df_processed.columns:
            st.error(f"Selected ID column '{id_column}' not found in the data.")
            st.stop()
        # Rename the selected ID column to 'result_code' for consistency
        df_processed = df_processed.rename(columns={id_column: 'result_code'})
    else:
        # Fall back to looking for common ID column names
        common_id_columns = ['Result code', 'result_code', 'ID', 'id', 'Code', 'code']
        found_id_column = None
        for col in common_id_columns:
            if col in df_processed.columns:
                found_id_column = col
                break
        
        if found_id_column:
            if found_id_column != 'result_code':
                df_processed = df_processed.rename(columns={found_id_column: 'result_code'})
        else:
            st.error("No ID column specified and no common ID columns found. Please select an ID column.")
            st.stop()
    
    if combine_evidence:
        try:
            df_processed = combine_rows_by_result_code(df_processed, selected_columns)  # Pass selected_columns
        except ValueError as e:
            st.error(str(e))
            st.stop()

    # Handle column selection for creating input_text
    if selected_columns:
        # Use the provided selected columns
        available_columns = [col for col in selected_columns if col in df_processed.columns]
        if not available_columns:
            st.error(f"None of the selected columns {selected_columns} are present in the data.")
            st.stop()
    else:
        # Fall back to the original hardcoded columns for backward compatibility
        possible_columns = ['Title', 'Description', 'Evidence Abstract Text', 'Evidence Parsed Text']
        available_columns = [col for col in possible_columns if col in df_processed.columns]
        
        if not available_columns:
            st.error("None of the expected text columns ('Title', 'Description', 'Evidence Abstract Text', 'Evidence Parsed Text') are present in the data. At least one is required to create 'input_text'.")
            st.stop()

    # Concatenate available text fields
    df_processed['input_text'] = df_processed[available_columns].fillna('').agg(' '.join, axis=1)
    return df_processed
