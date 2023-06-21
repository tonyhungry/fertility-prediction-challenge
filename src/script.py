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

python script.py input_path 

An example for the provided test is:

python script.py data/test_data_liss_2_subjects.csv
"""

import csv
import sys
import argparse
import pandas as pd

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

import pandas as pd

data = pd.read_csv("datasets/LISS_example_input_data.csv", encoding='cp1252', low_memory=False)
outcome = pd.read_csv("datasets/LISS_example_groundtruth_data.csv")

selected_columns = ['leeftijd2019','lftdcat2019', 'lftdhhh2019', 'aantalki2019', 'partner2019', 'burgstat2019', 'woonvorm2019',
                   'woning2019', 'belbezig2019', 'brutoink2019', 'nettoink2019', 'brutocat2019', 'nettocat2019',
                   'oplzon2019', 'oplmet2019', 'oplcat2019', 'sted2019', 'werving2019']
features = data[selected_columns]

y_isna = outcome['new_child'].isnull()
X_isna = features.isnull().any(axis=1)
features = features.drop(features[y_isna | X_isna].index)
outcome = outcome.drop(outcome[y_isna | X_isna].index)

from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(features)
categorical_columns = categorical_columns_selector(features)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, outcome['new_child'], test_size=0.30, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))

model.fit(X_train, y_train)
model.score(X_test, y_test)

from sklearn.model_selection import cross_validate
cv_result = cross_validate(model, features, outcome['new_child'], cv=5)
cv_result
    df["prediction"] = df["year"] % 2
    
    return df[["nomem_encr", "prediction"]]


def predict(input_path, output):
    if output is None:
        output = sys.stdout
    df = pd.read_csv(input_path)
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
    merged_df = pd.merge(predictions_df, ground_truth_df, on="nomem_encr")

    # Calculate accuracy
    accuracy = len(
        merged_df[merged_df["prediction"] == merged_df["outcome"]]
    ) / len(merged_df)

    # Calculate true positives, false positives, and false negatives
    true_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["outcome"] == 1)]
    )
    false_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["outcome"] == 0)]
    )
    false_negatives = len(
        merged_df[(merged_df["prediction"] == 0) & (merged_df["outcome"] == 1)]
    )

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
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



