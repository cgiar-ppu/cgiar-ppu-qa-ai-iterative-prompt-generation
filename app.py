# app.py

import streamlit as st
import pandas as pd
import os
import time
from data_loader import load_data, load_existing_results
from task_generator import generate_task_list, save_task_list
from executor import execute_tasks_concurrently
from evaluator import evaluate_results
from logger import ResultLogger
from response_extractor import add_score_explanation_columns
import config
import prompt_techniques
from datetime import datetime
import base64

# app.py
#Test Comment to refresh

# Sidebar
st.sidebar.title("Configuration")

# Model Selection
models = config.MODELS
selected_models = st.sidebar.multiselect("Select Models", models, default=models)

# Prompt Selection
prompts = config.PROMPTS

# Initialize available prompts
available_prompts = list(prompts.keys())

# Ensure selected prompts in session_state are valid
if 'selected_prompts' in st.session_state:
    # Remove any prompts that are no longer available
    st.session_state.selected_prompts = [p for p in st.session_state.selected_prompts if p in available_prompts]
else:
    # Initialize selected_prompts with all available prompts
    st.session_state.selected_prompts = available_prompts.copy()

# Display existing prompts with a fixed key to manage state
st.sidebar.subheader("Existing Prompts")
selected_prompts = st.sidebar.multiselect(
    "Select Prompts to Use",
    options=available_prompts,
    default=None,  # Default is None because we're using session_state
    key='selected_prompts'  # Use a fixed key to manage state
)

# Update session state with the current selection
# This happens automatically because of the key parameter

# Add New Prompts
st.sidebar.subheader("Add New Prompt")
with st.sidebar.expander("Add a New Prompt"):
    new_prompt_id = st.text_input("Prompt ID")
    new_prompt_text = st.text_area("Prompt Text")
    new_prompt_impact_area = st.selectbox(
        "Impact Area", ["Gender", "Nutrition", "Climate", "Environment", "Poverty", "IPSR", "None"]
    )
    add_prompt_button = st.button("Add Prompt")
    if add_prompt_button:
        if new_prompt_id and new_prompt_text:
            # Add the new prompt to the prompts dictionary
            prompts[new_prompt_id] = {
                'id': new_prompt_id,
                'text': new_prompt_text + " **Text to Analyze:** [INPUT_TEXT]",
                'impact_area': new_prompt_impact_area,
                'active': True
            }
            st.success(f"Prompt '{new_prompt_id}' added.")

            # Update available prompts
            available_prompts.append(new_prompt_id)

            # Add new prompt to the session state's selected_prompts
            st.session_state.selected_prompts.append(new_prompt_id)
        else:
            st.error("Please provide both Prompt ID and Prompt Text.")

# Upload Prompts from Excel
st.sidebar.subheader("Upload Prompts from Excel")
uploaded_prompt_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
if uploaded_prompt_file:
    try:
        prompt_df = pd.read_excel(uploaded_prompt_file)
        if all(col in prompt_df.columns for col in ["Prompt ID", "Prompt Text", "Impact Area"]):
            for _, row in prompt_df.iterrows():
                prompt_id = row["Prompt ID"]
                prompt_text = row["Prompt Text"]
                impact_area = row["Impact Area"]
                if prompt_id and prompt_text and impact_area:
                    prompts[prompt_id] = {
                        'id': prompt_id,
                        'text': prompt_text + " **Text to Analyze:** [INPUT_TEXT]",
                        'impact_area': impact_area,
                        'active': True
                    }
                    if prompt_id not in available_prompts:
                        available_prompts.append(prompt_id)
                    if prompt_id not in st.session_state.selected_prompts:
                        st.session_state.selected_prompts.append(prompt_id)
            st.success("Prompts from Excel file added successfully.")
        else:
            st.error("Excel file must contain 'Prompt ID', 'Prompt Text', and 'Impact Area' columns.")
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

# Dataset Selection
st.sidebar.subheader("Dataset")
dataset_option = st.sidebar.radio(
    "Select Dataset",
    ('Default Dataset', 'Upload Your Own')
)

if dataset_option == 'Upload Your Own':
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    input_df = None
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
else:
    input_csv = 'input/Joined_Processed_Evidence_PRMS_ExpertsScore.csv'
    input_df = load_data(input_csv)

if input_df is None:
    st.warning("Please upload a CSV file to proceed.")

# Result Limit
if input_df is not None:
    result_limit = st.sidebar.number_input(
        "Number of Results to Process",
        min_value=1,
        max_value=len(input_df),
        value=min(100, len(input_df))
    )
else:
    st.sidebar.warning("Please upload a CSV file to set the number of results to process.")

# Start Processing Button
start_button = st.sidebar.button("Start Processing")

def get_table_download_link(df, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">{link_text}</a>'
    return href

# app.py

# Function to update progress
def update_progress(n):
    total_tasks = len(tasks)
    progress_bar.progress(n / total_tasks)
    status_text.text(f"Processing task {n} of {total_tasks}")

if start_button:
    # Filter prompts
    selected_prompts_dict = {pid: prompts[pid] for pid in selected_prompts}
    
    # Deduplicate input data based on 'Result code' and relevant text fields
    df_unique_input = input_df.drop_duplicates(
        subset=['Result code', 'Title', 'Description', 'Evidence_Abstract_Text', 'Evidence_Parsed_Text']
    )
    
    # Limit the number of results
    df_unique_input = df_unique_input.head(int(result_limit))
    
    # Generate tasks
    tasks = generate_task_list(df_unique_input, selected_prompts_dict, selected_models)
    
    # Save task list
    task_list_csv = f'output//task_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv'
    task_list_excel = f'output//task_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx'
    save_task_list(tasks, task_list_excel, task_list_csv)

    # Display number of tasks
    st.write(f"Total tasks to process: {len(tasks)}")
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Execute tasks with progress update
    logger = ResultLogger(f"output/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    

    
# Execute tasks concurrently with progress update
    results = execute_tasks_concurrently(tasks, max_workers=8, progress_callback=update_progress)
    
    progress_bar.empty()
    status_text.text("Processing completed.")
    
    # Extract responses
    results_df = pd.DataFrame(results)
    results_df = add_score_explanation_columns(results_df, 'model_output')
    
    # Save results
    results_csv = f"output/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_csv, index=False)
    
    # Evaluate results
    metrics_csv = f"output/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    evaluate_results(results_csv, input_csv, metrics_csv)
    
    # Load metrics
    try:
        metrics_df = pd.read_csv(metrics_csv)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.write("No Metrics possible based on dataset provided")
        metrics_df = pd.DataFrame()

    input_df = pd.read_csv(input_csv)

    tasks_df = pd.read_excel(task_list_excel)

    # Provide download links
    st.subheader("Download Files")
 
    st.markdown(get_table_download_link(metrics_df, 'Download Metrics CSV'), unsafe_allow_html=True)
    st.markdown(get_table_download_link(results_df, 'Download Results CSV'), unsafe_allow_html=True)


    
    # Display metrics
    st.subheader("Metrics")
    st.dataframe(metrics_df)
    
    st.subheader("Outputs")
    st.dataframe(results_df)

    st.subheader("Full details sent to LLM")
    st.dataframe(tasks_df)

    st.subheader("Input used")
    st.dataframe(input_df)
