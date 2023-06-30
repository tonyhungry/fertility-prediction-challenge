"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the predict_outcomes function. 

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

The script can be run from the command line using the following command:

python script.py input_path .

An example for the provided test is:

python script.py data/test_data_liss_2_subjects.csv
"""

import os
import sys
import argparse
import pandas as pd
from joblib import load

parser = argparse.ArgumentParser(description="Process and score data.")
subparsers = parser.add_subparsers(dest="command")

# Process subcommand
process_parser = subparsers.add_parser("predict", help="Process input data for prediction.")
process_parser.add_argument("input_path", help="Path to input data CSV file.")
process_parser.add_argument("--output", help="Path to prediction output CSV file.")

# Score subcommand
score_parser = subparsers.add_parser("score", help="Score (evaluate) predictions.")
score_parser.add_argument("prediction_path", help="Path to predicted outcome CSV file.")
score_parser.add_argument("ground_truth_path", help="Path to ground truth outcome CSV file.")
score_parser.add_argument("--output", help="Path to evaluation score output CSV file.")

args = parser.parse_args()


def predict_outcomes(df):
    """Process the input data and write the predictions."""
    # Keep 
    results = df[["nomem_encr"]]
        
    # Select predictors: education, year of birth, gender, number of children in the household 
    # You can do this automatically (not necessarily better): https://scikit-learn.org/stable/modules/feature_selection.html
    keepcols = ['leeftijd2019','aantalki2019','partner2019','burgstat2019',
            'woonvorm2019','woning2019','belbezig2019',
            'oplmet2019','sted2019','brutohh_f2019','geslacht',
            'ch19l004','ch19l018', 'ch19l021', 'ch19l022', 'ch19l126','ch19l133',
            'ch19l159','ch19l160', 'ch19l161','ch19l162',
            'ch19l163', 'ch19l229','cp19k010','cp19k026','cv19k012','cv19k053',
            'cv19k101','cv19k125','cv19k126','cv19k130',
            'cv19k140','cr19l089','cr19l134','cs19l079','cs19l105','cs19l436',
            'cs19l435',
            'cf19l014','cf19l025','cf14g034','cf19l136','cf19l131','cf19l129',
            'cf19l130','cf19l133','cf19l134', 'cf19l128','cf14g035','cf19l013',
            'cf19l024','cf19l026','cf19l027','cf19l030','cf19l032','cf19l033',
            'cf19l034','cf19l068','cf19l212','cf19l252','cf19l453','cf19l504',
            'cf19l505','cf19l506','cf19l508','cv19k111','cf19l011',
            'cf19l183','cf19l198']
    
    
    df = df.loc[:, keepcols]

    # Dictionary used
    dict_kids = {'None': 0, 'One child': 1, 'Two children': 2, 'Three children': 3, 'Four children': 4, 'Five children': 5, 'Six children': 6}
    df["aantalki2019"] = df["aantalki2019"].map(dict_kids)
                            
    # Load your trained model from the models directory
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
    model = load(model_path)

    # Use your trained model for prediction
    results.loc[:, "prediction"] = model.predict(df)

    #If you use predict_proba to get a probability and a different threshold
    #df["prediction"] = (df["prediction"] >= 0.5).astype(int)
    return results


def predict(input_path, output):
    if output is None:
        output = sys.stdout
    df = pd.read_csv(input_path, encoding="latin-1", encoding_errors="replace", low_memory=False)
    predictions = predict_outcomes(df)
    assert (
        predictions.shape[1] == 2
    ), "Predictions must have two columns: nomem_encr and prediction"
    # Check for the columns, order does not matter
    assert set(predictions.columns) == set(
        ["nomem_encr", "prediction"]
    ), "Predictions must have two columns: nomem_encr and prediction"

    predictions.to_csv(output, index=False)


def score(prediction_path, ground_truth_path, output):
    """Score (evaluate) the predictions and write the metrics.
    
    This function takes the path to a CSV file containing predicted outcomes and the
    path to a CSV file containing the ground truth outcomes. It calculates the overall 
    prediction accuracy, and precision, recall, and F1 score for having a child 
    and writes these scores to a new output CSV file.

    This function should not be modified.
    """

    if output is None:
        output = sys.stdout
    # Load predictions and ground truth into dataframes
    predictions_df = pd.read_csv(prediction_path)
    ground_truth_df = pd.read_csv(ground_truth_path)

    # Merge predictions and ground truth on the 'id' column
    merged_df = pd.merge(predictions_df, ground_truth_df, on="nomem_encr", how="right")

    # Calculate accuracy
    accuracy = len(
        merged_df[merged_df["prediction"] == merged_df["new_child"]]
    ) / len(merged_df)

    # Calculate true positives, false positives, and false negatives
    true_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 1)]
    )
    false_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 0)]
    )
    false_negatives = len(
        merged_df[(merged_df["prediction"] == 0) & (merged_df["new_child"] == 1)]
    )

    # Calculate precision, recall, and F1 score
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0
    # Write metric output to a new CSV file
    metrics_df = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1_score]
    })
    metrics_df.to_csv(output, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.input_path, args.output)
    elif args.command == "score":
        score(args.prediction_path, args.ground_truth_path, args.output)
    else:
        parser.print_help()
        predict(args.input_path, args.output)  
        sys.exit(1)
