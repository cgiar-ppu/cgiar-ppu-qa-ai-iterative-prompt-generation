# output_conversion_unpivot_dashboard.py

import pandas as pd
from datetime import datetime

def transform_for_dashboard(df):
    """
    Transforms the input DataFrame for dashboard compatibility.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to transform.

    Returns:
    - pd.DataFrame: The transformed DataFrame.
    """
    # Pivot the DataFrame
    pivot_df = df.pivot_table(
        index='result_code',
        columns='impact_area',
        values='score',
        aggfunc='first'  # In case there are duplicates
    ).reset_index()

    # Flatten the columns (in case impact_area values are multi-level)
    pivot_df.columns = [col if isinstance(col, str) else col[1] if col[1] else col[0] for col in pivot_df.columns]

    return pivot_df
