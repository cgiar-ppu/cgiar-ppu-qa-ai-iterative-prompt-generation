# evaluator.py

import pandas as pd

def evaluate_results(output_csv, input_csv, metrics_csv, result_code_to_impact_areas):
    """
    Compare model outputs to expected values and calculate metrics.
    """

    # Load the results and input data
    results_df = pd.read_csv(output_csv)
    input_df = pd.read_csv(input_csv)

    # Map 'result_code' to 'Impact Area checked' in the results DataFrame
    results_df['Impact Area checked'] = results_df['result_code'].map(result_code_to_impact_areas)

    # Explode the 'Impact Area checked' list into separate rows
    results_df = results_df.explode('Impact Area checked')

    # Ensure the 'score' column exists in results_df
    if 'score' not in results_df.columns:
        raise ValueError("The 'score' column is missing in the results CSV. Please run response_extractor.py first.")

    # Preprocess columns for matching
    results_df['result_code'] = results_df['result_code'].astype(str).str.strip()
    results_df['Impact Area checked'] = results_df['Impact Area checked'].str.lower().str.strip()
    results_df['prompt_id'] = results_df['prompt_id'].astype(str).str.strip()
    # No need to process 'impact_area' here since we're not using it for merging

    input_df['Result code'] = input_df['Result code'].astype(str).str.strip()
    input_df['Impact Area checked'] = input_df['Impact Area checked'].astype(str).str.strip().str.lower()


    # Merge on 'result_code' and 'Impact Area'
    merged_df = pd.merge(
        results_df,
        input_df,
        left_on=['result_code', 'Impact Area checked'],
        right_on=['Result code', 'Impact Area checked'],
        how='inner'
    )

    print(f"Merged DataFrame shape: {merged_df.shape}")

    if merged_df.empty:
        print("Warning: The merged DataFrame is empty. Check if 'result_code' and 'Impact Area' values match between the results and input data.")
        return  # Exit the function early

    # Ensure 'Expert score' and 'score' are numeric
    merged_df['Expert score'] = pd.to_numeric(merged_df['Expert score'], errors='coerce')
    merged_df['score'] = pd.to_numeric(merged_df['score'], errors='coerce')

    # Drop rows with NaN scores
    merged_df = merged_df.dropna(subset=['score', 'Expert score'])

    # Create a column to indicate if the model's score matches the expert score
    merged_df['correct'] = merged_df['score'] == merged_df['Expert score']

    # Calculate metrics
    metrics = []

    # Group by model_name and Impact Area
    grouped = merged_df.groupby(['model_name', 'Impact Area checked', 'prompt_id'])

    for (model_name, impact_area_checked, prompt_id), group in grouped:
        total = len(group)
        correct = group['correct'].sum()
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 0

        # Breakdown by score value
        score_breakdown = group.groupby('Expert score')['correct'].agg(['sum', 'count'])
        score_metrics = []
        for expert_score, row in score_breakdown.iterrows():
            score_total = row['count']
            score_correct = row['sum']
            score_accuracy = score_correct / score_total if score_total > 0 else 0
            score_metrics.append({
                'expert_score': expert_score,
                'total': score_total,
                'correct': score_correct,
                'accuracy': score_accuracy
            })

        metrics.append({
            'model_name': model_name,
            'Impact Area': impact_area_checked,
            'prompt_id': prompt_id,
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy,
            'score_metrics': score_metrics
        })

    # Save metrics to a CSV file
    metrics_list = []
    for item in metrics:
        for score_metric in item['score_metrics']:
            metrics_list.append({
                'model_name': item['model_name'],
                'Impact Area': item['Impact Area'],
                'prompt_id': item['prompt_id'],
                'total': item['total'],
                'correct': item['correct'],
                'incorrect': item['incorrect'],
                'accuracy': item['accuracy'],
                'expert_score': score_metric['expert_score'],
                'score_total': score_metric['total'],
                'score_correct': score_metric['correct'],
                'score_accuracy': score_metric['accuracy']
            })

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics saved to {metrics_csv}")
