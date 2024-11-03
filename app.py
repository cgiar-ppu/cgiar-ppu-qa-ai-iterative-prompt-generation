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

# Sidebar
st.sidebar.title("Configuration")

# Model Selection
models = config.MODELS
selected_models = st.sidebar.multiselect("Select Models", models, default=models)

# Prompt Selection
prompts = config.PROMPTS

# Display existing prompts
st.sidebar.subheader("Existing Prompts")
selected_prompts = st.sidebar.multiselect(
    "Select Prompts to Use",
    list(prompts.keys()),
    default=list(prompts.keys())
)

# Add New Prompts
st.sidebar.subheader("Add New Prompt")
with st.sidebar.expander("Add a New Prompt"):
    new_prompt_id = st.text_input("Prompt ID")
    new_prompt_text = st.text_area("Prompt Text")
    new_prompt_impact_area = st.text_input("Impact Area")
    add_prompt_button = st.button("Add Prompt")
    if add_prompt_button:
        if new_prompt_id and new_prompt_text:
            prompts[new_prompt_id] = {
                'id': new_prompt_id,
                'text': new_prompt_text,
                'impact_area': new_prompt_impact_area,
                'active': True
            }
            st.success(f"Prompt '{new_prompt_id}' added.")
            selected_prompts.append(new_prompt_id)
        else:
            st.error("Please provide both Prompt ID and Prompt Text.")

# Dataset Selection
st.sidebar.subheader("Dataset")
dataset_option = st.sidebar.radio(
    "Select Dataset",
    ('Default Dataset', 'Upload Your Own')
)

if dataset_option == 'Upload Your Own':
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
else:
    input_csv = 'input/Joined_Processed_Evidence_PRMS_ExpertsScore.csv'
    input_df = load_data(input_csv)

# Result Limit
result_limit = st.sidebar.number_input(
    "Number of Results to Process",
    min_value=1,
    max_value=len(input_df),
    value=min(100, len(input_df))
)

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
    metrics_df = pd.read_csv(metrics_csv)
    
    # Display metrics
    st.subheader("Metrics")
    st.dataframe(metrics_df)
    
    # Provide download links
    st.subheader("Download Files")
    st.markdown(get_table_download_link(results_df, 'Download Results CSV'), unsafe_allow_html=True)
    st.markdown(get_table_download_link(metrics_df, 'Download Metrics CSV'), unsafe_allow_html=True)




