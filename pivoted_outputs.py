import pandas as pd

def create_pivoted_outputs(input_df, results_df, id_column='result_code'):
    # Ensure the results_df contains the selected id column
    if id_column != 'result_code' and id_column not in results_df.columns:
        results_df = results_df.copy()
        results_df[id_column] = results_df['result_code']

    # Deduplicate input by the selected identifier to ensure 1 row per unique ID
    input_df_dedup = input_df.drop_duplicates(subset=[id_column]).copy()

    pivoted = results_df.pivot_table(
        index=[id_column, 'model_name'],
        columns='prompt_id',
        values='model_output',
        aggfunc='first'
    ).reset_index()

    merged = pd.merge(input_df_dedup, pivoted, on=id_column, how='left')

    # Drop the internal alias if it's different from the selected id to avoid confusion in the export
    if id_column != 'result_code' and 'result_code' in merged.columns:
        merged = merged.drop(columns=['result_code'])

    return merged