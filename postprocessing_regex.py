"""RegEx confirmation layer for the results of LLM extraction. 
This code checks if the values extracted by the LLM are actually present in the 'Conclusões' text, 
and if not, it replaces them with NaN."""

import pandas as pd
import numpy as np
import re
import os 


def value_in_conclusoes(text, val):
    #If no value is present, skip verification
    if pd.isna(val):
        return False
    
    #Convert to float
    try:
        val = float(val)
    except:
        return False

    # Extract all numbers from text
    numbers_in_text = re.findall(r'\d+[.,]?\d*', str(text))
    numbers_in_text = [float(n.replace(',', '.')) for n in numbers_in_text]

    # Check if val is exactly in numbers_in_text
    is_there = val in numbers_in_text
    
    #If not present, print to analyse what is happening
    if is_there == False:
        print(text)
        print(numbers_in_text)
        print(val)

    return is_there


#Folders
results_folder = f"results/<model_name_folder>"
output_folder = f"{results_folder}_confirmed"

#Create a output folder for the results after RegEx confirmation
if not os.path.exists(f"{output_folder}"):
    os.makedirs(f"{output_folder}")
else:
    print(f"Folder {output_folder} already exists. Results may be overwritten.")
    
#Load the results from LLM extraction
if os.path.exists(f"{results_folder}/extraction_results.xlsx"):
    df = pd.read_excel(f"{results_folder}/extraction_results.xlsx")


#Columns to confirm
ffr_ifr_cols = [
    'Tronco_Comum_FFR',
    'Descendente_Anterior_FFR',
    'Circunflexa_FFR',
    'Coronária_Direita_FFR',
    'Outras_artérias_FFR',
    'Tronco_Comum_iFR',
    'Descendente_Anterior_iFR',
    'Circunflexa_iFR',
    'Coronária_Direita_iFR',
    'Outras_artérias_iFR'
]

df_filtered = df.copy()
    
for col in ffr_ifr_cols:
    df_filtered[col] = df_filtered.apply(
        lambda row: row[col] if value_in_conclusoes(row['Conclusões'], row[col]) else np.nan,
        axis=1)

df_filtered.to_excel(f"{output_folder}/extraction_results.xlsx", index=False)