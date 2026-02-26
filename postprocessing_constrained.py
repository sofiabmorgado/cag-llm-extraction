"""Format the extracted results using the constrained pipeline."""

import pandas as pd
import json
import numpy as np


def safe_json_load(x):
    # If it's already NaN / None → return NaN
    if pd.isna(x):
        return np.nan
    
    # If it's not a string, we also treat it as invalid
    if not isinstance(x, str):
        return np.nan

    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return np.nan


model_folder = "<model_name_folder>" 
df = pd.read_excel(f"results/{model_folder}/extraction.xlsx")
df = df[['id', 'Conclusões','extracted']]

df_json = df['extracted'].apply(safe_json_load)
df_final = pd.json_normalize(df_json)

# Save the result to a new Excel file
df_final.to_excel(f"results/{model_folder}/extraction_results.xlsx", index=False)
