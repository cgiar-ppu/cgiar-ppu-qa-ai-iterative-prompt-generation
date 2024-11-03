import re
import pandas as pd

# Define regex patterns to capture both quoted and unquoted score values
pattern1 = r'```json\s*\{\s*"score":\s*"?([^",]+)"?,\s*"explanation":\s*"([^"]+)"'
pattern2 = r'\{\s*"score":\s*"?([^",]+)"?,\s*"explanation":\s*"([^"]+)"'

# Function to extract score and explanation
def extract_values(text):
    match = re.search(pattern1, text) or re.search(pattern2, text)
    return (match.group(1), match.group(2)) if match else (None, None)

# Function to add score and explanation columns to DataFrame
def add_score_explanation_columns(df, column_name):
    # Apply the extraction function and split into two new columns
    df[['score', 'explanation']] = df[column_name].apply(extract_values).apply(pd.Series)
    return df
