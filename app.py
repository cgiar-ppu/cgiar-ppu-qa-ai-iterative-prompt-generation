# app.py
import io
import streamlit as st
import pandas as pd
import os
import time
from data_loader import load_data, load_existing_results
from task_generator import generate_task_list, save_task_list
from executor import execute_tasks_concurrently
from executor import client
from evaluator import evaluate_results
from logger import ResultLogger
from response_extractor import add_score_explanation_columns
import config
import prompt_techniques
from datetime import datetime
import base64
import re
import json  # <--- NEW IMPORT for parsing JSON from second call output
from output_conversion_unpivot_dashboard import transform_for_dashboard  # Import the transformation function

# >>>> NEW IMPORT <<<<
import critic # this is for the prompt critic
import improver  # this is for the prompt improver
import traceback

HELP_TEXT = """
## Overview

This application helps you to:
- **Manage and run multiple ChatGPT prompts** against a dataset of research results.
- **Generate bulk evaluations** using different prompts and models.
- **Upload your own data** and apply prompts to produce aggregated, downloadable outputs.
- **Transform outputs** for custom dashboards or further analysis.
- **Perform follow-up interactions** with previously generated model responses.

The intended workflow is:
1. **Configure Models & Prompts**
2. **Select or Upload a Dataset**
3. **Choose Results to Process**
4. **Run the Main Processing to Generate Outputs**
5. **Evaluate Metrics and Download Results**
6. **If Needed, Perform Follow-up Prompts**

Below you will find detailed instructions for each step.

---

## Step-by-Step Guide

### 1. Configuration

- **Select Models:**  
  In the sidebar, choose one or multiple available models to run your prompts.  
  The list of models appears under "Configuration" in the left sidebar.  
  You can select multiple models to run tasks in parallel.

- **Select Prompts:**  
  Under "Configuration" → "Existing Prompts," choose which prompts you want the model(s) to process.  
  You may also **Add New Prompts** manually or by uploading an Excel file with prompt definitions.  
  Prompts must include the placeholder `[INPUT_TEXT]`, which the app will replace with your dataset content.

**Tip:** Ensure each prompt is relevant and well-defined since it will guide the model's response and scoring.

### 2. Dataset Selection

- **Default Dataset vs. Upload Your Own:**  
  In the sidebar, select **"Default Dataset"** to use the pre-loaded sample data or **"Upload Your Own"** to provide a CSV/Excel file.  
  If uploading, ensure your file has relevant text fields (e.g., Title, Description, Evidence_Abstract_Text, or Evidence_Parsed_Text).  
  Once uploaded, the system automatically concatenates these fields into `input_text`.

**Tip:** Expand the "Show sample of input data" section in the main page to confirm correct columns and data formatting.

### 3. Selecting Results to Process

- **By Number of Results:**  
  Choose how many top rows of your dataset you want to process.  
  This is useful for a quick test or smaller subsets.

- **By Specific Result Codes:**  
  Paste a list of result codes to filter. This approach allows targeted evaluation of specific entries from your dataset.  
  The app confirms the entered codes and shows how many matches were found.

**Tip:** Make sure that the "Result code" column in your dataset matches what you enter.

### 4. Running the Main Processing

- Once configurations, prompts, and results are set, click **"Start Processing"** in the sidebar.  
- The app will generate a list of tasks (each task = a combination of a result code, a prompt, and a model).  
- It runs these tasks concurrently and displays progress as they complete.  
- After completion, you'll see:
  - Download links for metrics and results.
  - A transformed dataset for dashboard analysis.
  - Detailed tables showing metrics, outputs, and the full tasks executed.

**Tip:** If no tasks are generated or the results are empty, re-check your prompt IDs, dataset, or filters.

### 5. Interpreting and Downloading Results

- **Metrics CSV:**  
  Provides aggregated evaluations comparing model scores to expert benchmarks (if provided in the dataset).
  
- **Results CSV:**  
  Contains raw model outputs for each task, including assigned scores and explanations.
  
- **Transformed Results for Dashboard (Excel):**  
  A pivoted format designed for easy integration into visualization tools.

**Tip:** Use the "Download" links to store results locally for offline analysis.

### 6. Custom Dashboard Transformation

- Under "Custom Dashboard Transformation" in the sidebar, you can upload a previously modified results CSV.
- Click "Transform Custom Results for Dashboard" to pivot and reshape it for dashboard-ready format.
- After transformation, a new download link will appear in the main page.

**Tip:** This step allows you to reuse existing results with a different layout.

### 7. Follow-up Prompts

- After completing the main processing, switch to the "Follow-up Prompts" tab.  
- Select a model, a prompt ID, and a specific result code to view the initial conversation context (input text and initial model response).  
- Enter a new follow-up prompt to delve deeper into that specific result.  
- The app displays the entire conversation, allowing iterative refinement and exploration.

**Tip:** The follow-up prompts are for deeper analysis or clarifications on already processed items.

---

## Managing Prompts

- **Adding Prompts Manually:**  
  Enter a "Prompt ID," the "Prompt Text," and select an "Impact Area."  
  The prompt text must contain `[INPUT_TEXT]` to indicate where the dataset's text should be inserted, or the app will append the placeholder automatically.
  
- **Uploading Prompts from Excel:**  
  Provide an Excel file with columns: "Prompt ID", "Prompt Text", "Impact Area".  
  All imported prompts become available for selection.  
  Use this feature to load multiple prompts quickly.

---

## Troubleshooting

- **No Data or Missing Columns:**  
  Check that your uploaded file contains at least one of the recognized text fields.
  
- **No Matching Result Codes:**  
  Ensure correct formatting of codes. Try uppercase and verify that they exist in the dataset.

- **Empty Results or Missing Metrics:**  
  If no tasks are generated or results are empty, verify that prompts are properly selected and that `[INPUT_TEXT]` was found in the prompt.  
  If metrics are not generated, it may be because the dataset lacks the columns required for benchmark comparisons.

---

## Additional Tips

- **Experiment with a Small Subset First:**  
  Start with a small number of results to ensure your prompts and dataset are working as intended.
  
- **Refine Prompts:**  
  Prompts can significantly affect the model's output. Adjust them for clarity and specificity, then re-run.

- **Keep a Record:**  
  Save downloaded results and metrics for reference and comparison across different runs.

---

We hope this guide helps you navigate the tool effectively. If you encounter issues, review the steps or refine your data and prompts for better results.
"""

simplified_models = ['o1-preview', 'o1-mini']

# Initialize session state variables
if 'results_df' not in st.session_state:
    st.session_state['results_df'] = pd.DataFrame()

if 'tasks_df' not in st.session_state:
    st.session_state['tasks_df'] = pd.DataFrame()

if 'custom_df' not in st.session_state:
    st.session_state['custom_df'] = None

if 'transformed_custom_df' not in st.session_state:
    st.session_state['transformed_custom_df'] = None

# We'll store the accuracy from the "Improve Prompt" process here:
if 'improver_accuracy' not in st.session_state:
    st.session_state['improver_accuracy'] = None

# We'll also store the actual prompt text and the model used, so we can display them in the table
if 'improver_prompt_text' not in st.session_state:
    st.session_state['improver_prompt_text'] = None

if 'improver_model_used' not in st.session_state:
    st.session_state['improver_model_used'] = None

# We'll store the DataFrame from the "Improve Prompt" result (iteration=1)
if 'improver_df' not in st.session_state:
    st.session_state['improver_df'] = pd.DataFrame()

# >>>> ADDED THIS TO MOVE THE FULL API RESPONSE TO THE IMPROVER TAB <<<<
if 'improver_full_response' not in st.session_state:
    st.session_state['improver_full_response'] = None


# ======================= SIDEBAR SECTION =======================
st.sidebar.title("Configuration")

# Model Selection (multi-select)
models = config.MODELS
selected_models = st.sidebar.multiselect(
    "Select Models",
    options=models,
    default=models
)

# Prompt Selection (multi-select)
prompts = config.PROMPTS
available_prompts = list(prompts.keys())

if 'selected_prompts' in st.session_state:
    st.session_state.selected_prompts = [
        p for p in st.session_state.selected_prompts if p in available_prompts
    ]
else:
    st.session_state.selected_prompts = available_prompts.copy()

st.sidebar.subheader("Existing Prompts")
selected_prompts = st.sidebar.multiselect(
    "Select Prompts to Use",
    options=available_prompts,
    default=None,
    key='selected_prompts'
)

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
            if "[INPUT_TEXT]" in new_prompt_text:
                prompt_text = new_prompt_text.replace("[INPUT_TEXT]", "[INPUT_TEXT]")
            else:
                prompt_text = new_prompt_text + " **Text to Analyze:** [INPUT_TEXT]"

            prompts[new_prompt_id] = {
                'id': new_prompt_id,
                'text': prompt_text,
                'impact_area': new_prompt_impact_area,
                'active': True
            }
            st.success(f"Prompt '{new_prompt_id}' added.")

            available_prompts.append(new_prompt_id)
            st.session_state.selected_prompts.append(new_prompt_id)
        else:
            st.error("Please provide both Prompt ID and Prompt Text.")

# Upload Prompts from Excel
st.sidebar.subheader("Upload Prompts from Excel")
uploaded_prompt_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
if uploaded_prompt_file:
    try:
        prompt_df = pd.read_excel(uploaded_prompt_file)
        required_cols = ["Prompt ID", "Prompt Text", "Impact Area"]
        if all(col in prompt_df.columns for col in required_cols):
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
            st.error(f"Excel file must contain {required_cols} columns.")
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

# Prompt Improvement Section
st.sidebar.subheader("Prompt Improvement")
selected_prompt_to_improve = st.sidebar.selectbox(
    "Prompt to be Improved",
    options=available_prompts,  # all prompts, independent from the multi-select
    key="prompt_to_improve"
)
selected_model_to_improve = st.sidebar.selectbox(
    "Model to Improve Prompt",
    options=models,  # all models, independent from the multi-select
    key="model_to_improve"
)
num_attempts = st.sidebar.number_input(
    "Number of Attempts",
    min_value=1,
    max_value=20,
    value=2,
    step=1,
    key="num_attempts_for_improvement"
)
improve_prompt_button = st.sidebar.button("Improve Prompt")

# Dataset Selection
st.sidebar.subheader("Dataset")
dataset_option = st.sidebar.radio(
    "Select Dataset",
    ('Default Dataset', 'Upload Your Own')
)

if dataset_option == 'Upload Your Own':
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])
    input_df = None
    if uploaded_file:
        input_df = load_data(uploaded_file)
else:
    input_file = 'input/Joined_Processed_Evidence_PRMS_ExpertsScore.csv'
    input_df = load_data(input_file)

# If no data
if input_df is None:
    st.warning("Please upload a CSV file to proceed.")
else:
    with st.expander("Show sample of input data for confirmation:"):
        st.write("Columns in DataFrame after processing:", input_df.columns.tolist())
        st.write("Sample data:")
        st.write(input_df.head())

# Result Selection Method
st.sidebar.subheader("Select Results")
result_selection_method = st.sidebar.radio(
    "Select Results by",
    ('Number of Results', 'Result Codes')
)

selected_df = pd.DataFrame()

if input_df is not None:
    if result_selection_method == 'Number of Results':
        result_limit = st.sidebar.number_input(
            "Number of Results to Process",
            min_value=1,
            max_value=len(input_df),
            value=min(100, len(input_df))
        )
        selected_df = input_df.head(int(result_limit))
    elif result_selection_method == 'Result Codes':
        result_codes = st.sidebar.text_area(
            "Paste Result Codes (comma, space, tab, or newline separated)",
            placeholder="e.g., RC001, RC002, RC003 or RC001 RC002 RC003"
        )
        if result_codes:
            result_code_list = [code.strip() for code in re.split(r'[,\s]+', result_codes) if code.strip()]
            with st.expander("Show confirmation of result codes entered by user:"):
                st.write("Result codes entered by user:")
                st.write(result_code_list)

            input_df['Result code'] = input_df['Result code'].astype(str).str.strip()
            input_df['Result code'] = input_df['Result code'].str.upper()
            result_code_list = [code.upper() for code in result_code_list]
            selected_df = input_df[input_df['Result code'].isin(result_code_list)]

            if selected_df.empty:
                st.warning("No matching result codes found. Please check your input.")
        else:
            selected_df = pd.DataFrame()
else:
    st.sidebar.warning("Please upload a CSV file to set the number of results to process.")

if selected_df.empty:
    st.warning("No results selected. Please adjust your selection criteria.")

start_button = st.sidebar.button("Start Processing")

# Custom Dashboard Transformation
st.sidebar.subheader("Custom Dashboard Transformation")
uploaded_custom_csv = st.sidebar.file_uploader("Upload Modified Results CSV for Dashboard Transformation", type=['csv'], key='custom_dashboard_upload')
if uploaded_custom_csv is not None:
    try:
        custom_df = pd.read_csv(uploaded_custom_csv)
        st.session_state['custom_df'] = custom_df
        st.sidebar.success("File uploaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error processing the uploaded custom CSV file: {e}")
else:
    if 'custom_df' in st.session_state and st.session_state['custom_df'] is not None:
        custom_df = st.session_state['custom_df']

transform_custom_button = st.sidebar.button("Transform Custom Results for Dashboard", key='transform_custom_button')
if transform_custom_button:
    if 'custom_df' in st.session_state and st.session_state['custom_df'] is not None:
        try:
            transformed_custom_df = transform_for_dashboard(st.session_state['custom_df'])
            st.session_state['transformed_custom_df'] = transformed_custom_df
            st.sidebar.success("Transformation completed. Please check the main page for the download link.")
        except Exception as e:
            st.sidebar.error(f"Error during transformation: {e}")
    else:
        st.sidebar.warning("Please upload a custom CSV file first.")


# ====================== HELPER FUNCTIONS ======================
def get_table_download_link(df, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="data.csv">{link_text}</a>'

def get_excel_download_link(df, link_text):
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data.xlsx">{link_text}</a>'

def update_progress(completed_tasks, total_tasks):
    progress_bar.progress(completed_tasks / total_tasks)
    status_text.text(f"Processing task {completed_tasks} of {total_tasks}")

def run_processing(df_input, prompt_dict, model_list):
    """
    1) Generate tasks
    2) Execute them
    3) Save results
    4) Evaluate results
    5) Return results_df, metrics_df, tasks_df
    """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    task_list_csv = f'output//task_list_{timestamp}.csv'
    task_list_excel = f'output//task_list_{timestamp}.xlsx'
    results_csv = f"output/results_{timestamp}.csv"
    metrics_csv = f"output/metrics_{timestamp}.csv"

    task_list = generate_task_list(df_input, prompt_dict, model_list)
    if not task_list:
        st.warning("No tasks generated. Please check your inputs.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    save_task_list(task_list, task_list_excel, task_list_csv)

    st.write(f"Total tasks to process: {len(task_list)}")

    global progress_bar, status_text
    progress_bar = st.progress(0)
    status_text = st.empty()

    logger = ResultLogger(results_csv)

    def callback_fn(completed_count):
        update_progress(completed_count, len(task_list))

    results = execute_tasks_concurrently(
        task_list, max_workers=8, progress_callback=callback_fn
    )

    progress_bar.empty()
    status_text.text("Processing completed.")

    results_df_local = pd.DataFrame(results)
    if results_df_local.empty:
        st.warning("No results were returned from processing.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    results_df_local = add_score_explanation_columns(results_df_local, 'model_output')
    results_df_local.to_csv(results_csv, index=False)

    if dataset_option == 'Upload Your Own':
        input_csv = f"output/uploaded_input_{timestamp}.csv"
        df_input.to_csv(input_csv, index=False)
    else:
        input_csv = input_file

    evaluate_results(results_csv, input_csv, metrics_csv)

    try:
        metrics_df_local = pd.read_csv(metrics_csv)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.write("No Metrics possible based on dataset provided")
        metrics_df_local = pd.DataFrame()

    tasks_df_local = pd.read_excel(task_list_excel)

    return results_df_local, metrics_df_local, tasks_df_local


# ======================== CREATE TABS FIRST ========================
help_tab, main_processing_tab, followup_tab, improver_tab = st.tabs([
    "Help", "Main Processing", "Follow-up Prompts", "Improver Results"
])

# ------------------------ HELP TAB ------------------------
with help_tab:
    st.title("How to Use This Application")
    st.markdown(HELP_TEXT)

# ------------------------ MAIN PROCESSING TAB ------------------------
with main_processing_tab:

    # 1) START PROCESSING
    if start_button:
        if selected_df.empty:
            st.warning("No results selected. Please adjust your selection criteria.")
        elif not selected_models:
            st.warning("No models selected. Please select at least one model.")
        elif not selected_prompts:
            st.warning("No prompts selected. Please select at least one prompt.")
        else:
            optional_columns = ['Title', 'Description', 'Evidence_Abstract_Text', 'Evidence_Parsed_Text']
            existing_columns = [col for col in optional_columns if col in selected_df.columns]
            df_unique_input = selected_df.drop_duplicates(subset=['Result code'] + existing_columns)

            if result_selection_method == 'Number of Results':
                df_unique_input = df_unique_input.head(int(result_limit))

            if df_unique_input.empty:
                st.warning("No unique input data to process after deduplication.")
            else:
                # Use all selected prompts and all selected models
                selected_prompts_dict = {pid: prompts[pid] for pid in selected_prompts}
                results_df_main, metrics_df_main, tasks_df_main = run_processing(
                    df_unique_input,
                    selected_prompts_dict,
                    selected_models
                )

                if not results_df_main.empty:
                    st.session_state['results_df'] = results_df_main
                    st.session_state['tasks_df'] = tasks_df_main

                    st.subheader("Download Files")
                    st.markdown(
                        get_table_download_link(metrics_df_main, 'Download Metrics CSV'),
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        get_table_download_link(results_df_main, 'Download Results CSV'),
                        unsafe_allow_html=True
                    )
                    transformed_results_df_main = transform_for_dashboard(results_df_main)
                    st.markdown(
                        get_excel_download_link(transformed_results_df_main, 'Download Results for dashboard in Excel'),
                        unsafe_allow_html=True
                    )

                    st.subheader("Metrics")
                    st.dataframe(metrics_df_main)

                    st.subheader("Outputs")
                    st.dataframe(results_df_main)

                    st.subheader("Full details sent to LLM")
                    st.dataframe(tasks_df_main)

                    st.subheader("Input used")
                    st.dataframe(input_df)

    # 2) IMPROVE PROMPT
    if improve_prompt_button:
        if selected_df.empty:
            st.warning("No results selected. Please adjust your selection criteria.")
        elif not selected_model_to_improve:
            st.warning("No model selected in 'Model to Improve Prompt.'")
        elif not selected_prompt_to_improve:
            st.warning("No prompt selected in 'Prompt to be Improved.'")
        else:
            optional_columns = ['Title', 'Description', 'Evidence_Abstract_Text', 'Evidence_Parsed_Text']
            existing_columns = [col for col in optional_columns if col in selected_df.columns]
            df_unique_input = selected_df.drop_duplicates(subset=['Result code'] + existing_columns)

            if result_selection_method == 'Number of Results':
                df_unique_input = df_unique_input.head(int(result_limit))

            if df_unique_input.empty:
                st.warning("No unique input data to process after deduplication.")
            else:
                # Use only the single prompt from "selected_prompt_to_improve"
                single_prompt_dict = {
                    selected_prompt_to_improve: prompts[selected_prompt_to_improve]
                }

                # Use only the single model from "Model to Improve Prompt"
                single_model_list = [selected_model_to_improve]

                # Store the text of the prompt used for improvement so we can display it later
                st.session_state['improver_prompt_text'] = prompts[selected_prompt_to_improve]['text']
                st.session_state['improver_prompt_area'] = prompts[selected_prompt_to_improve]['impact_area']
                # Store the model used as well
                st.session_state['improver_model_used'] = selected_model_to_improve

                # --------------- FIRST API CALL - TO GET INITIAL PROMPT AND ACCURACY ---------------
                results_df_imp, metrics_df_imp, tasks_df_imp = run_processing(
                    df_unique_input,
                    single_prompt_dict,
                    selected_models
                    #single_model_list
                )

                st.write("Initial Metrics Calculator DataFrame")
                st.dataframe(metrics_df_imp)
                st.write("Initial Results Calculator DataFrame")
                st.dataframe(results_df_imp)

                # Build and store a more granular accuracy object, if metrics are available
                if not metrics_df_imp.empty:
                    label_accuracy_list = []
                    for _, row in metrics_df_imp.iterrows():
                        label_accuracy_list.append({
                            "gold_label": row["expert_score"],
                            "total_score_results": row["score_total"],
                            "correct_model_pred_results": row["score_correct"],
                            "accuracy": row["score_accuracy"]
                        })
                    general_accuracy = metrics_df_imp['accuracy'].iloc[0] if 'accuracy' in metrics_df_imp.columns else None
                    st.session_state['improver_accuracy'] = {
                        "general_accuracy": general_accuracy,
                        "label_accuracy": label_accuracy_list
                    }
                else:
                    st.session_state['improver_accuracy'] = None

                # --------------- BUILD INITIAL DATAFRAME TABLE (iteration=1) ---------------
                iteration_val = 1
                prompt_title_val = st.session_state['prompt_to_improve']
                prompt_val = st.session_state['improver_prompt_text'] or ""
                model_val = st.session_state['improver_model_used'] or ""
                accuracy_val = st.session_state['improver_accuracy']
                changes_val = ""

                improver_data = {
                    "iteration": [iteration_val],
                    "prompt_title": [prompt_title_val],
                    "prompt": [prompt_val],
                    "model": [model_val],
                    "accuracy": [accuracy_val],
                    "changes": [changes_val]
                }
                improver_df_local = pd.DataFrame(improver_data)
                st.session_state['improver_df'] = improver_df_local

                # Loop through the number of attempts (2A through 2C)
                for iteration_val in range(2, num_attempts + 1):

                    # --------------- SECOND API CALL - 2A - TO GET THE OBSERVATIONS FROM THE CRITIC MODEL ---------------
                    # Make sure each dataframe's "result_code" or "Result code" is a string:
                    results_df_imp["result_code"] = results_df_imp["result_code"].astype(str).str.strip()
                    tasks_df_imp["result_code"] = tasks_df_imp["result_code"].astype(str).str.strip()
                    input_df["Result code"] = input_df["Result code"].astype(str).str.strip()

                    # Also ensure "impact_area" and "Impact Area checked" match in case format:
                    results_df_imp["impact_area"] = results_df_imp["impact_area"].astype(str).str.strip().str.lower()
                    input_df["Impact Area checked"] = input_df["Impact Area checked"].astype(str).str.strip().str.lower()

                    # Build the partial model_out dataframe:
                    df_model_out = results_df_imp[["result_code", "impact_area", "score", "explanation"]].copy()
                    df_model_out.rename(columns={
                        "explanation": "model_rationale",
                        "score": "model_label"
                    }, inplace=True)

                    # Build the partial input_for_critic dataframe:
                    df_input_for_critic = input_df[["Result code", "Impact Area checked", "Expert score"]].copy()

                    # Merge on (result_code, impact_area) = (Result code, Impact Area checked)
                    df_merged_tmp = pd.merge(
                        df_model_out,
                        df_input_for_critic,
                        left_on=["result_code", "impact_area"],
                        right_on=["Result code", "Impact Area checked"],
                        how="inner"
                    )

                    # Drop rows missing model_label or Expert score if you want only valid pairs
                    df_merged_tmp = df_merged_tmp.dropna(subset=["model_label", "Expert score"])

                    # Merge in the "input_text" from tasks_df_imp
                    df_input_details = tasks_df_imp[["result_code", "input_text"]].copy()
                    df_input_details["result_code"] = df_input_details["result_code"].astype(str).str.strip()

                    df_merged_2 = pd.merge(
                        df_merged_tmp,
                        df_input_details,
                        on="result_code",
                        how="left"
                    )

                    df_merged_2.rename(
                        columns={"Expert score": "gold_label"},
                        inplace=True
                    )

                    # Ensure model_label is numeric
                    df_merged_2['model_label'] = df_merged_2['model_label'].astype(int)

                    # Also ensure gold_label is numeric if needed
                    df_merged_2['gold_label'] = df_merged_2['gold_label'].astype(int)

                    st.write("Merged DataFrame for Critic")
                    st.dataframe(df_merged_2)
                    st.write(df_merged_2.dtypes)

                    import random
                    evaluation_data_examples = []
                    for gl in [2, 1, 0]:
                        for ml in [2, 1, 0]:
                            subset = df_merged_2[
                                (df_merged_2["gold_label"] == gl) & (df_merged_2["model_label"] == ml)
                            ]
                            if not subset.empty:
                                row = subset.sample(n=1).iloc[0]
                                evaluation_data_examples.append({
                                    "input_text": row.get("input_text", ""),
                                    "gold_label": int(gl),
                                    "model_label": int(ml),
                                    "model_rationale": row.get("model_rationale", "")
                                })
                            else:
                                evaluation_data_examples.append({
                                    "gold_label": int(gl),
                                    "model_label": int(ml),
                                    "model_rationale": "No examples found for this combination."
                                })

                    critic_payload = {
                        "prompt_text": prompt_val,
                        "evaluation_data_examples": evaluation_data_examples
                    }
                    critic_payload_str = json.dumps(critic_payload, indent=2)
                    st.write("critic_user_content")
                    st.json(critic_payload_str)

                    critic_messages = []

                    critic_system_content = critic.PROMPTS['PROMPT_CRITIC']['text']
                    if selected_model_to_improve not in simplified_models:
                        critic_messages.append({"role": "system", "content": critic_system_content})

                    critic_user_content = f"Here is the JSON with the prompt and examples to be evaluated:\n\n{critic_payload_str}"
                    critic_messages.append({"role": "user", "content": critic_user_content})

                    try:
                        if selected_model_to_improve in ['o1-mini', 'o1']:
                            critic_response = client.chat.completions.create(
                                model=selected_model_to_improve,
                                messages=critic_messages,
                                max_completion_tokens=15000
                            )
                        else:
                            critic_response = client.chat.completions.create(
                                model=selected_model_to_improve,
                                messages=critic_messages,
                                temperature=0,
                                max_tokens=15000,
                                top_p=0,
                                frequency_penalty=0,
                                presence_penalty=0
                            )
                        st.session_state['critic_full_response'] = critic_response
                    except Exception as e:
                        st.error(f"Error querying OpenAI for iteration {iteration_val}: {e}")
                        st.error(traceback.format_exc())
                        break

                    isolated_critic_content = st.session_state['critic_full_response'].choices[0].message.content
                    # Regex to extract the JSON object if inside triple backticks
                    critic_json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                    critic_match = re.search(critic_json_pattern, isolated_critic_content, re.DOTALL)
                    if critic_match:
                        # Grab the JSON inside the code fence
                        critic_response_json_str = critic_match.group(1)
                    else:
                        # If nothing was captured by the regex, assume the whole thing is JSON
                        critic_response_json_str = isolated_critic_content

                    parsed_critic_json = json.loads(critic_response_json_str)
                    critic_model_recommendations = parsed_critic_json.get("proposed_improvements", "")
                    critic_model_recommendations_str = json.dumps(critic_model_recommendations)
                    st.session_state['critic_model_recommendations'] = critic_model_recommendations_str
                    st.write('critic_model_recommendations')
                    st.write(st.session_state['critic_model_recommendations'])

                    # >>>> ADD NEW COLUMN AND STORE CRITIC RECOMMENDATIONS IN THE PREVIOUS ITERATION ROW <<<<
                    if 'critic_recommendations' not in st.session_state['improver_df'].columns:
                        st.session_state['improver_df']['critic_recommendations'] = ""
                    st.session_state['improver_df'].loc[
                        st.session_state['improver_df']['iteration'] == iteration_val - 1,
                        'critic_recommendations'
                    ] = critic_model_recommendations_str

                    # --------------- THIRD API CALL - 2B - TO GET EDITED PROMPT FROM THE IMPROVER MODEL ---------------
                    improver_df_local = st.session_state['improver_df']
                    df_str = improver_df_local.to_json(orient='records')
                    df_json = json.loads(df_str)

                    st.json(df_json)

                    improver_prompt_text = improver.PROMPTS['PROMPT_IMPROVER']['text']
                    # Prepare the messages
                    improver_messages = []
                    if selected_model_to_improve not in simplified_models:
                        # The system message is the prompt text contained in improver.py
                        improver_system_content = improver_prompt_text
                        improver_messages.append({"role": "system", "content": improver_system_content})

                    # The user content is only the JSON with the previous prompts and additional details. The "improver prompt" is sent as the system message.
                    improver_user_content = f"Here is the JSON with the previous prompts and additional details:\n\n{df_str}"
                    improver_messages.append({"role": "user", "content": improver_user_content})


                    try:
                        if selected_model_to_improve in ['o1-mini', 'o1']:
                            response = client.chat.completions.create(
                                model=selected_model_to_improve,
                                messages=improver_messages,
                                max_completion_tokens=15000
                            )
                        else:
                            response = client.chat.completions.create(
                                model=selected_model_to_improve,
                                messages=improver_messages,
                                temperature=0,
                                max_tokens=15000,
                                top_p=0,
                                frequency_penalty=0,
                                presence_penalty=0
                            )

                        # >>>> STORE THE FULL API RESPONSE IN SESSION STATE <<<<
                        st.session_state['improver_full_response'] = response
                        print(response)

                        isolated_content = st.session_state['improver_full_response'].choices[0].message.content

                        # Regex to find a JSON object between triple backticks (with or without "json" after them)
                        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                        match = re.search(json_pattern, isolated_content, re.DOTALL)
                        if match:
                            # Grab the JSON inside the code fence
                            improver_response_json_str = match.group(1)
                        else:
                            # If nothing was captured by the regex, assume the whole thing is JSON
                            improver_response_json_str = isolated_content

                        parsed_json = json.loads(improver_response_json_str)
                        revised_prompt_value = parsed_json.get("revised_prompt", "")
                        changes_value = parsed_json.get("changes", "")
                        st.session_state['latest_improved_prompt'] = revised_prompt_value
                        st.session_state['latest_changes'] = changes_value

                        # --------------- ADD ROW (iteration) TO THE DF ---------------
                        improved_prompt_title = st.session_state['prompt_to_improve'] + "_v" + str(iteration_val)
                        prompt_val_2 = revised_prompt_value
                        model_val_2 = st.session_state['improver_model_used']
                        accuracy_val_2 = ""
                        changes_val_2 = changes_value

                        new_row = pd.DataFrame({
                            "iteration": [iteration_val],
                            "prompt_title": [improved_prompt_title],
                            "prompt": [prompt_val_2],
                            "model": [model_val_2],
                            "accuracy": [accuracy_val_2],
                            "changes": [changes_val_2]
                        })

                        st.session_state['improver_df'] = pd.concat(
                            [st.session_state['improver_df'], new_row],
                            ignore_index=True
                        )

                    except Exception as e:
                        st.error(f"Error querying OpenAI for iteration {iteration_val}: {e}")
                        st.error(traceback.format_exc())
                        break

                    # --------------- FOURTH API CALL - 2C - TO GET THE ACCURACY FOR THE EDITED PROMPT ---------------
                    prompts[improved_prompt_title] = {
                        'id': improved_prompt_title,
                        'text': prompt_val_2,
                        'impact_area': st.session_state['improver_prompt_area'],
                        'active': True
                    }

                    single_prompt_dict = {
                        selected_prompt_to_improve: prompts[improved_prompt_title]
                    }

                    results_df_imp, metrics_df_imp, tasks_df_imp = run_processing(
                        df_unique_input,
                        single_prompt_dict,
                        selected_models
                        #single_model_list
                    )

                    st.write("Subsequent Metrics Calculator Dataframe")
                    st.dataframe(metrics_df_imp)
                    st.write("Subsequent Results Calculator Dataframe")
                    st.dataframe(results_df_imp)

                    # Build and store a more granular accuracy object, if metrics are available
                    if not metrics_df_imp.empty:
                        label_accuracy_list = []
                        for _, row in metrics_df_imp.iterrows():
                            label_accuracy_list.append({
                                "gold_label": row["expert_score"],
                                "total_score_results": row["score_total"],
                                "correct_model_pred_results": row["score_correct"],
                                "accuracy": row["score_accuracy"]
                            })
                        general_accuracy = metrics_df_imp['accuracy'].iloc[0] if 'accuracy' in metrics_df_imp.columns else None
                        st.session_state['improver_accuracy'] = {
                            "general_accuracy": general_accuracy,
                            "label_accuracy": label_accuracy_list
                        }
                    else:
                        st.session_state['improver_accuracy'] = None

                    accuracy_json = json.dumps(st.session_state['improver_accuracy'], default=str)

                    # Now this is just a single string, so it can be assigned to a single cell:
                    st.session_state['improver_df'].iloc[-1, st.session_state['improver_df'].columns.get_loc("accuracy")] = accuracy_json

    # If the transformed data is available, display the download link
    if 'transformed_custom_df' in st.session_state and st.session_state['transformed_custom_df'] is not None:
        transformed_custom_df = st.session_state['transformed_custom_df']
        st.subheader("Transformed Dashboard Data")
        st.markdown(
            get_excel_download_link(transformed_custom_df, 'Download Transformed Dashboard Excel'),
            unsafe_allow_html=True
        )

# ------------------------ FOLLOW-UP PROMPTS TAB ------------------------
with followup_tab:
    if 'results_df' not in st.session_state or 'tasks_df' not in st.session_state:
        st.warning("No results or tasks available for follow-up prompts. Please run the main processing first.")
    else:
        results_df = st.session_state['results_df']
        tasks_df = st.session_state['tasks_df']
        if results_df.empty or tasks_df.empty:
            st.warning("No results or tasks to proceed with follow-up prompts.")
        else:
            st.subheader("Follow-up Prompts")

            selected_model = st.selectbox(
                "Select Model for Follow-up",
                options=results_df['model_name'].unique(),
                key='followup_model'
            )
            selected_prompt_id = st.selectbox(
                "Select Prompt ID for Follow-up",
                options=results_df['prompt_id'].unique(),
                key='followup_prompt_id'
            )
            selected_result_code = st.selectbox(
                "Select Result Code for Follow-up",
                options=results_df['result_code'].unique(),
                key='followup_result_code'
            )
            followup_prompt = st.text_area("Enter your follow-up prompt", key='followup_prompt')
            submit_followup = st.button("Submit Follow-up Prompt")

            if submit_followup and followup_prompt:
                conv_key = f"conversation_{selected_model}_{selected_prompt_id}_{selected_result_code}"

                if conv_key not in st.session_state:
                    st.session_state[conv_key] = []
                    try:
                        input_text_row = tasks_df.loc[
                            (tasks_df['result_code'] == selected_result_code) &
                            (tasks_df['prompt_id'] == selected_prompt_id) &
                            (tasks_df['model_name'] == selected_model)
                        ]

                        if input_text_row.empty:
                            st.error("No matching task found for the selected combination.")
                            st.stop()

                        input_text = input_text_row['input_text'].iloc[0]
                        prompt_text_template = prompts[selected_prompt_id]['text']
                        initial_prompt_text = prompt_text_template.replace('[INPUT_TEXT]', input_text)

                        initial_response_row = results_df.loc[
                            (results_df['result_code'] == selected_result_code) &
                            (results_df['prompt_id'] == selected_prompt_id) &
                            (results_df['model_name'] == selected_model)
                        ]

                        if initial_response_row.empty:
                            st.error("No initial response found for the selected combination.")
                            st.stop()

                        initial_response = initial_response_row['model_output'].iloc[0]

                        # For non-simplified models, include a system role
                        initial_messages = []
                        if selected_model not in simplified_models:
                            role = (
                                "You are an assistant that will closely follow the instruction provided next "
                                "and respond in a concise way by providing a direct answer and also a very brief "
                                "explanation without too many details."
                            )
                            initial_messages.append({"role": "system", "content": role})

                        initial_messages.append({"role": "user", "content": initial_prompt_text})
                        initial_messages.append({"role": "assistant", "content": initial_response})
                        st.session_state[conv_key] = initial_messages

                    except Exception as e:
                        st.error(f"Error retrieving initial conversation: {e}")
                        st.stop()

                st.session_state[conv_key].append({"role": "user", "content": followup_prompt})
                messages = st.session_state[conv_key]
                try:
                    if selected_model in simplified_models:
                        response = client.chat.completions.create(
                            model=selected_model,
                            messages=messages
                        )
                    else:
                        response = client.chat.completions.create(
                            model=selected_model,
                            messages=messages,
                            temperature=0,
                            max_tokens=15000,
                            top_p=0,
                            frequency_penalty=0,
                            presence_penalty=0
                        )

                    output = response.choices[0].message.content.strip()
                    st.session_state[conv_key].append({"role": "assistant", "content": output})

                    st.subheader("Conversation")
                    conversation = st.session_state[conv_key]
                    for msg in conversation:
                        if msg['role'] == 'user':
                            st.markdown(f"**User:** {msg['content']}")
                        elif msg['role'] == 'assistant':
                            st.markdown(f"**Assistant:** {msg['content']}")
                        elif msg['role'] == 'system':
                            pass

                except Exception as e:
                    st.error(f"Error querying OpenAI: {e}")

# ------------------------ IMPROVER RESULTS TAB ------------------------
with improver_tab:
    st.subheader("Improver Results")
    # Display the improver_df if we have data
    if not st.session_state['improver_df'].empty:
        st.dataframe(st.session_state['improver_df'])
        
    else:
        st.write("The prompt improvement process has not been run yet, or there is no data.")
