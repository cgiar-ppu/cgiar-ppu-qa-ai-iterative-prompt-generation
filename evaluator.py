# evaluator.py

import pandas as pd
import json
import re

def parse_model_output(model_output):
    """
    Parse the model output to extract the score and explanation.
    """
    try:
        # Try parsing as JSON
        parsed_output = json.loads(model_output)
        score = parsed_output.get('score')
        explanation = parsed_output.get('explanation')
    except json.JSONDecodeError:
        # Fallback to regex parsing
        score_match = re.search(r'"score":\s*(\d)', model_output)
        explanation_match = re.search(r'"explanation":\s*"([^"]+)"', model_output)
        score = int(score_match.group(1)) if score_match else None
        explanation = explanation_match.group(1) if explanation_match else None
    return score, explanation

def evaluate_results(results_df, input_df):
    """
    Compare model outputs to expected values and calculate metrics.
    """
    metrics = []
    for _, row in results_df.iterrows():
        score, explanation = parse_model_output(row['model_output'])
        expected_column = perspective_to_column(row['prompt_id'])
        if expected_column and expected_column in input_df.columns:
            expected_value = input_df.loc[input_df['Result code'] == row['result_code'], expected_column].values[0]
            correct = int(score) == int(expected_value)
            metrics.append({
                'result_code': row['result_code'],
                'prompt_id': row['prompt_id'],
                'model_name': row['model_name'],
                'correct': correct
            })
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

def perspective_to_column(prompt_id):
    """
    Map prompt IDs to the corresponding performance column.
    """
    mapping = {
        'GENDER_EQUALITY_PROMPT': 'Gender level',
        'CLIMATE_CHANGE_PROMPT': 'Climate change level',
        'NUTRITION_PROMPT': 'Nutrition tag level',
        'ENVIRONMENTAL_HEALTH_PROMPT': 'Environmental biodiversity tag level',
        'POVERTY_REDUCTION_PROMPT': 'Poverty reduction tag level',
        'INNOVATION_READINESS_PROMPT': 'IPSR LEVEL',
        # GEOSCOPE_TAG_PROMPT and ACTOR_VERIFICATION_PROMPT have no evaluation
    }
    return mapping.get(prompt_id, None)