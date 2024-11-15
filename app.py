# app.py
import io
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
import re
from output_conversion_unpivot_dashboard import transform_for_dashboard  # Import the transformation function
# app.py
#Test Comment to refresh x2
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
            # Check if "[INPUT_TEXT]" is in new_prompt_text
            if "[INPUT_TEXT]" in new_prompt_text:
                prompt_text = new_prompt_text.replace("[INPUT_TEXT]", "[INPUT_TEXT]")
            else:
                prompt_text = new_prompt_text + " **Text to Analyze:** [INPUT_TEXT]"

            # Add the new prompt to the prompts dictionary
            prompts[new_prompt_id] = {
                'id': new_prompt_id,
                'text': prompt_text,
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
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])
    input_df = None
    if uploaded_file:
        # Use load_data function to process the uploaded file
        input_df = load_data(uploaded_file)
else:
    input_file = 'input/Joined_Processed_Evidence_PRMS_ExpertsScore.csv'
    input_df = load_data(input_file)
    #with st.expander("Show Debug Logs"):    
       # st.write("Columns in DataFrame:", input_df.columns.tolist())
       # st.write("Available Result Codes in DataFrame:")
       # st.write(input_df['Result code'].unique())

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

selected_df = pd.DataFrame()  # Initialize selected_df

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
            "Paste Result Codes (separated by comma, space, tab, or newline)",
            placeholder="e.g., RC001, RC002, RC003 or RC001 RC002 RC003"
        )
        if result_codes:
            # Split by any whitespace or comma
            result_code_list = [code.strip() for code in re.split(r'[,\s]+', result_codes) if code.strip()]
            with st.expander("Show confirmation of result codes entered by user:"):
                st.write("Result codes entered by user:")
                st.write(result_code_list)

            # Ensure 'Result code' column is of type string and strip whitespace
            input_df['Result code'] = input_df['Result code'].astype(str).str.strip()

            # Convert both to uppercase for case-insensitive matching
            input_df['Result code'] = input_df['Result code'].str.upper()
            result_code_list = [code.upper() for code in result_code_list]

            # Perform the filtering
            selected_df = input_df[input_df['Result code'].isin(result_code_list)]

            if selected_df.empty:
                st.warning("No matching result codes found. Please check your input.")
        else:
            selected_df = pd.DataFrame()  # Empty DataFrame if no codes are provided
else:
    st.sidebar.warning("Please upload a CSV file to set the number of results to process.")

if selected_df.empty:
    st.warning("No results selected. Please adjust your selection criteria.")

# Start Processing Button
start_button = st.sidebar.button("Start Processing")

def get_table_download_link(df, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">{link_text}</a>'
    return href

def get_excel_download_link(df, link_text):
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, header=True)
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data.xlsx">{link_text}</a>'
    return href

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

# Button to Transform Custom Results
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

# Function to update progress
def update_progress(n):
    total_tasks = len(tasks)
    progress_bar.progress(n / total_tasks)
    status_text.text(f"Processing task {n} of {total_tasks}")

# Create Tabs
tab1, tab2 = st.tabs(["Main Processing", "Follow-up Prompts"])

with tab1:
    if start_button:
        # Check for empty selections
        if selected_df.empty:
            st.warning("No results selected. Please adjust your selection criteria.")
        elif not selected_models:
            st.warning("No models selected. Please select at least one model.")
        elif not selected_prompts:
            st.warning("No prompts selected. Please select at least one prompt.")
        else:
            # Filter prompts
            selected_prompts_dict = {pid: prompts[pid] for pid in selected_prompts}

            # Deduplicate input data based on 'Result code' and relevant text fields
            optional_columns = ['Title', 'Description', 'Evidence_Abstract_Text', 'Evidence_Parsed_Text']
            existing_columns = [col for col in optional_columns if col in selected_df.columns]
            df_unique_input = selected_df.drop_duplicates(subset=['Result code'] + existing_columns)

            # Limit the number of results if using 'Number of Results' method
            if result_selection_method == 'Number of Results':
                df_unique_input = df_unique_input.head(int(result_limit))

            # Check if df_unique_input is empty
            if df_unique_input.empty:
                st.warning("No unique input data to process after deduplication.")
            else:
                # Generate tasks
                tasks = generate_task_list(df_unique_input, selected_prompts_dict, selected_models)

                # Check if tasks list is empty
                if not tasks:
                    st.warning("No tasks generated. Please check your inputs.")
                else:
                    # Save task list
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    task_list_csv = f'output//task_list_{timestamp}.csv'
                    task_list_excel = f'output//task_list_{timestamp}.xlsx'
                    save_task_list(tasks, task_list_excel, task_list_csv)

                    # Display number of tasks
                    st.write(f"Total tasks to process: {len(tasks)}")

                    # Initialize progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Execute tasks with progress update
                    logger = ResultLogger(f"output/results_{timestamp}.csv")

                    # Execute tasks concurrently with progress update
                    results = execute_tasks_concurrently(tasks, max_workers=8, progress_callback=update_progress)

                    progress_bar.empty()
                    status_text.text("Processing completed.")

                    # Extract responses
                    results_df = pd.DataFrame(results)
                    st.session_state['results_df'] = results_df

                    if results_df.empty:
                        st.warning("No results were returned from processing.")
                    else:
                        results_df = add_score_explanation_columns(results_df, 'model_output')

                        # Save results
                        results_csv = f"output/results_{timestamp}.csv"
                        results_df.to_csv(results_csv, index=False)

                        # Define input_csv for evaluate_results
                        if dataset_option == 'Upload Your Own':
                            input_csv = f"output/uploaded_input_{timestamp}.csv"
                            input_df.to_csv(input_csv, index=False)
                        else:
                            input_csv = input_file
                            pass

                        # Evaluate results
                        metrics_csv = f"output/metrics_{timestamp}.csv"
                        evaluate_results(results_csv, input_csv, metrics_csv)

                        # Load metrics
                        try:
                            metrics_df = pd.read_csv(metrics_csv)
                        except (FileNotFoundError, pd.errors.EmptyDataError):
                            st.write("No Metrics possible based on dataset provided")
                            metrics_df = pd.DataFrame()

                        tasks_df = pd.read_excel(task_list_excel)
                        st.session_state['tasks_df'] = tasks_df

                        # Provide download links
                        st.subheader("Download Files")

                        st.markdown(get_table_download_link(metrics_df, 'Download Metrics CSV'), unsafe_allow_html=True)
                        st.markdown(get_table_download_link(results_df, 'Download Results CSV'), unsafe_allow_html=True)

                        # New button for downloading results in the desired format for the dashboard
                        transformed_results_df = transform_for_dashboard(results_df)
                        st.markdown(get_excel_download_link(transformed_results_df, 'Download Results for dashboard in Excel'), unsafe_allow_html=True)

                        # Display metrics
                        st.subheader("Metrics")
                        st.dataframe(metrics_df)

                        st.subheader("Outputs")
                        st.dataframe(results_df)

                        st.subheader("Full details sent to LLM")
                        st.dataframe(tasks_df)

                        st.subheader("Input used")
                        st.dataframe(input_df)

                        # Store critical data in session_state
                        st.session_state['results_df'] = results_df
                        st.session_state['tasks_df'] = tasks_df

    # If the transformed data is available, display the download link
    if 'transformed_custom_df' in st.session_state and st.session_state['transformed_custom_df'] is not None:
        transformed_custom_df = st.session_state['transformed_custom_df']
        st.subheader("Transformed Dashboard Data")
        st.markdown(get_excel_download_link(transformed_custom_df, 'Download Transformed Dashboard Excel'), unsafe_allow_html=True)

    # =================== NEW CODE FOR FOLLOW-UP PROMPTS ===================

    # ... [Previous code remains unchanged] ...

with tab2:
    # =================== UPDATED CODE FOR FOLLOW-UP PROMPTS ===================
    
    # Ensure tasks_df and results_df are available
    if 'results_df' not in st.session_state or 'tasks_df' not in st.session_state:
        st.warning("No results or tasks available for follow-up prompts. Please run the main processing first.")
    else:
        results_df = st.session_state['results_df']
        tasks_df = st.session_state['tasks_df']
        if results_df.empty or tasks_df.empty:
            st.warning("No results or tasks to proceed with follow-up prompts.")
        else:
            st.subheader("Follow-up Prompts")

            # Select Model
            selected_model = st.selectbox(
                "Select Model for Follow-up",
                options=results_df['model_name'].unique(),
                key='followup_model'
            )

            # Select Prompt ID
            selected_prompt_id = st.selectbox(
                "Select Prompt ID for Follow-up",
                options=results_df['prompt_id'].unique(),
                key='followup_prompt_id'
            )

            # Select Result Code
            selected_result_code = st.selectbox(
                "Select Result Code for Follow-up",
                options=results_df['result_code'].unique(),
                key='followup_result_code'
            )

            # Input Follow-up Prompt
            followup_prompt = st.text_area("Enter your follow-up prompt", key='followup_prompt')

            # Button to Submit Follow-up Prompt
            submit_followup = st.button("Submit Follow-up Prompt")

            if submit_followup and followup_prompt:
                # Build conversation history

                # Key for session state
                conv_key = f"conversation_{selected_model}_{selected_prompt_id}_{selected_result_code}"

                # Initialize conversation if not exists
                if conv_key not in st.session_state:
                    st.session_state[conv_key] = []

                    # Retrieve initial prompt and response
                    try:
                        # Get the input text for the result code
                        input_text_row = tasks_df.loc[
                            (tasks_df['result_code'] == selected_result_code) &
                            (tasks_df['prompt_id'] == selected_prompt_id) &
                            (tasks_df['model_name'] == selected_model)
                        ]

                        if input_text_row.empty:
                            st.error("No matching task found for the selected combination.")
                            # Optionally provide debug information
                            # st.write(tasks_df[['result_code', 'prompt_id', 'model_name']])
                            st.stop()

                        input_text = input_text_row['input_text'].iloc[0]
                        # Retrieve the prompt text from the prompts dictionary
                        prompt_text_template = prompts[selected_prompt_id]['text']
                        initial_prompt_text = prompt_text_template.replace('[INPUT_TEXT]', input_text)

                        # Get the initial model response
                        initial_response_row = results_df.loc[
                            (results_df['result_code'] == selected_result_code) &
                            (results_df['prompt_id'] == selected_prompt_id) &
                            (results_df['model_name'] == selected_model)
                        ]

                        if initial_response_row.empty:
                            st.error("No initial response found for the selected combination.")
                            # Optionally provide debug information
                            # st.write(results_df[['result_code', 'prompt_id', 'model_name']])
                            st.stop()

                        initial_response = initial_response_row['model_output'].iloc[0]

                        # For non-simplified models, include system role
                        initial_messages = []
                        if selected_model not in simplified_models:
                            role = "You are an assistant that will closely follow the instruction provided next and respond in a concise way by providing a direct answer and also a very brief explanation without too many details."
                            initial_messages.append({"role": "system", "content": role})

                        initial_messages.append({"role": "user", "content": initial_prompt_text})
                        initial_messages.append({"role": "assistant", "content": initial_response})

                        # Save to session state
                        st.session_state[conv_key] = initial_messages

                    except Exception as e:
                        st.error(f"Error retrieving initial conversation: {e}")
                        st.stop()

                # Append the follow-up prompt to the conversation
                st.session_state[conv_key].append({"role": "user", "content": followup_prompt})

                # Send the conversation to the model
                messages = st.session_state[conv_key]
                from executor import client
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
                            temperature=0,  # Adjust as needed
                            max_tokens=500,
                            top_p=0,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
                    output = response.choices[0].message.content.strip()

                    # Append the model's response to the conversation
                    st.session_state[conv_key].append({"role": "assistant", "content": output})

                    # Display the conversation
                    st.subheader("Conversation")
                    # Retrieve the conversation history
                    conversation = st.session_state[conv_key]

                    # Display the entire conversation
                    for msg in conversation:
                        if msg['role'] == 'user':
                            st.markdown(f"**User:** {msg['content']}")
                        elif msg['role'] == 'assistant':
                            st.markdown(f"**Assistant:** {msg['content']}")
                        elif msg['role'] == 'system':
                            pass  # Optionally display system messages

                except Exception as e:
                    st.error(f"Error querying OpenAI: {e}")
    # =================== END OF UPDATED CODE ===================