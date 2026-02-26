"""Code for evaluation of the results."""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sympy import true
import sys


def is_valid(val):
    if pd.isna(val):  # NaN is allowed
        return True
    try:
        num = float(val)
        return 0 <= num <= 100 #sometimes FFR/iFR can be slightly above 1, and in one case it is 84 meaning 0.84
    except:
        return False


def evaluate_binary(y_true, y_pred):
    """Evaluate binary classification"""
    # Print the 2-entry table for counts of values 0 and 1
    counts_table = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'])
    print(counts_table)
     
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred), 3)
    recall = round(recall_score(y_true, y_pred), 3)
    f1 = round(f1_score(y_true, y_pred), 3)
    
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(f1))


def evaluate_FFR_iFR(df_true, df_pred):
    """Evaluate the extraction of FFR and iFR in 3 phases:
    1. Evaluate the format: are the values numeric?
    2. Evaluate the presence/absence of values
    3. Evaluate the MSE for the values that are present in both DataFrames"""

    #Select columns for evaluation
    columns = ["Tronco_Comum_FFR", "Descendente_Anterior_FFR", "Circunflexa_FFR", "Coronária_Direita_FFR", "Outras_artérias_FFR", "Tronco_Comum_iFR", "Descendente_Anterior_iFR", "Circunflexa_iFR", "Coronária_Direita_iFR", "Outras_artérias_iFR"]
    df_pred = df_pred[columns]
    df_true = df_true[columns]
    
    #Replace "NA" with None to be able to count missing values
    df_pred = df_pred.replace("NA", None)

    #Number of missing values in each column
    missing = df_pred.isna().sum().sum()
    print(f"Missing: {missing}")
    
    #Number of values present
    total_values = df_pred.size - missing
    print(f"Number of extracted values: {total_values}")
    
    #How many where non-numeric?
    out_of_format_values = []

    for c in df_pred.columns:
        mask_invalid = ~df_pred[c].apply(is_valid)
        if mask_invalid.any():
            out_of_format_values.append(
                df_pred.loc[mask_invalid, c]
                .to_frame(name="value")
                .assign(column=c)
            )

    if out_of_format_values:
        out_of_format_df = pd.concat(out_of_format_values)
    else:
        out_of_format_df = pd.DataFrame(columns=["value", "column"])

    print(f"Out of format: {len(out_of_format_df)}")
    print(out_of_format_df)
    

    # Flatten the DataFrames and create binary presence/absence arrays
    print("Presence/Absence evaluation")
    true_presence = df_true.notna().values.flatten().astype(int)
    pred_presence = df_pred.notna().values.flatten().astype(int)
    evaluate_binary(true_presence, pred_presence)
    
    # Convert all values to floats (handle commas and NaNs)
    for df in [df_pred, df_true]:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.').str.strip(), errors='coerce')

    # Flatten the DataFrames for comparison
    pred_values = df_pred.values.flatten()
    true_values = df_true.values.flatten()

    # Mask to ignore NaNs in true values
    mask = ~np.isnan(pred_values) & ~np.isnan(true_values)

    # Compute accuracy: how many predictions match true values
    accuracy = np.mean(pred_values[mask] == true_values[mask])
    print(f"Accuracy: {accuracy}")


#RUN
results_folder = "<model_name_folder>"
df_true = pd.read_csv('data/reports_groundtruth.csv')
df_pred = pd.read_excel(f'results/{results_folder}/extraction_results.xlsx')
evaluate_FFR_iFR(df_true, df_pred)