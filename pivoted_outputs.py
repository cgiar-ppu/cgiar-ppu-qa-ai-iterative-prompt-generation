import pandas as pd

def create_pivoted_outputs(input_df, results_df):
    pivoted = results_df.pivot_table(
        index=['result_code', 'model_name'],
        columns='prompt_id',
        values='model_output',
        aggfunc='first'
    ).reset_index()
    merged = pd.merge(input_df, pivoted, on='result_code', how='left')
    return merged