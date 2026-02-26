"""Extraction with baseline RegEx."""

import re
import unicodedata
import pandas as pd
import os
import numpy as np
import regex
from sklearn.model_selection import train_test_split


def remove_accents(text):
    if isinstance(text, str):
        text = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in text if not unicodedata.combining(c))
    return text


def create_cols(df, tex_col, top_separators):
    top_separators = [remove_accents(x) for x in top_separators]
    separators = top_separators
    sep_pattern = '|'.join(regex.escape(sep) for sep in separators)
    pattern = rf'({sep_pattern})'

    def split_unordered(text):
    # splits text by separators if it does not find fills with "" 
        parts = regex.split(pattern, text)
        result = {sep: "" for sep in separators}

        current_sep = None
        for part in parts:
            part = part.strip()
            if (part in separators) and result[part] == "":
                current_sep = part
            elif current_sep:
                result[current_sep] += (" " + part)

        return pd.Series(result)

    split_df = df[tex_col].fillna('').astype(str).apply(remove_accents).apply(split_unordered)
    df_full = pd.concat([df, split_df], axis=1)
    df_full[top_separators] = df_full[top_separators].replace("", np.nan)
    return df_full


def structure_text_column(df, text_column, top_separators):
    print(f"\n Creating new columns from original column {text_column}")
    print(f"\n Separators to be used: {top_separators}")

    df = create_cols(df,text_column, top_separators)

    # fixing column names
    df.columns = [c.lower() if i >= len(df.columns) - len(top_separators) else c for i, c in enumerate(df.columns)]
    return df


def typo_correction(df):
    """Remove typos. we are not using this function now due to bias."""
    # remove typos
    typo_map = {
        "CORONARIOGAFIA": "CORONARIOGRAFIA",
        "CORONARIAGRAFIA": "CORONARIOGRAFIA",
        "COROANRIOGRAFIA": "CORONARIOGRAFIA"
    }

    for typo, correct in typo_map.items():
        df["clean_text"] = df["clean_text"].str.replace(typo, correct, case=False, regex=True)
    return df


def word_distance(text, start, end):
    """Count words between two character positions."""
    snippet = text[start:end]
    return len(re.findall(r'\w+', snippet))


def extract_vessel_measures(text, measure_keywords, vessel_keywords, MAX_WORD_DISTANCE, default_measure):
    # default_measure = if no measure name (iFR / FFR) is found defaults to this (chose iFR or FFR)

    # find all matches for numbers like 0.XX or 0,XX
    num_pattern = r'\b0[.,]\d+\b'
    matches = [(m.group(), m.start()) for m in re.finditer(num_pattern, text,re.IGNORECASE)]
    results = {}
    
    for num, pos in matches:

        # look only at text before match
        num_val = float(num.replace(',', '.'))
        before_text = text[:pos]
        
        # Find closest measure before number - case sensisitve
        nearest_measure, nearest_measure_pos = None, -1
        for measure in measure_keywords:
            p = before_text.rfind(measure)
            if p > nearest_measure_pos:
                nearest_measure = measure
                nearest_measure_pos = p
        
        # Find closest vessel before number - should be case insentitive
        nearest_vessel, nearest_vessel_pos = None, -1
        for vessel_name, vessel_key in vessel_keywords.items():
            for vessel in vessel_key:
                p = before_text.lower().rfind(vessel.lower())
                if p > nearest_vessel_pos:
                    nearest_vessel = vessel_name
                    nearest_vessel_pos = p
        
        # Calculate distances (in words)
        if nearest_measure_pos != -1:
            measure_distance = word_distance(text, nearest_measure_pos, pos)
        else:
            measure_distance = float('inf')
        
        if nearest_vessel_pos != -1:
            vessel_distance = word_distance(text, nearest_vessel_pos, pos)
        else:
            vessel_distance = float('inf')
        
        # Skip if both are too far
        if min(measure_distance, vessel_distance) > MAX_WORD_DISTANCE:
            continue
        
        # Create column name
        if nearest_vessel and vessel_distance <= MAX_WORD_DISTANCE:
            colname = f"{nearest_vessel.replace(' ', '_')}_{nearest_measure if nearest_measure else default_measure}"
        elif nearest_measure and measure_distance <= MAX_WORD_DISTANCE:
            colname = nearest_measure.upper()
        else:
            continue
        
        if results.get(colname) is None:
            results[colname] = [num_val,measure_distance,vessel_distance]
        else:
            # replace if it is closer to vessel than previous
            [n,md,vd] = results[colname]
            if vessel_distance < vd:
                results[colname] = [num_val,measure_distance,vessel_distance]

    # keep only the values
    for key, value in results.items():
        results[key] = value[0]
    return pd.Series(results)


def extract_stent_measures(text, measure_keywords, vessel_keywords, MAX_WORD_DISTANCE, default_measure):
    if not isinstance(text, str):
        return pd.Series()

    # Find all stent length patterns
    num_pattern = r'\b\d{1,3}\s*mm\b'
    matches = [(m.group(), m.start()) for m in re.finditer(num_pattern, text, re.IGNORECASE)]
    results = {}

    for num, pos in matches:

        # Extract only the number before "mm"
        num_val = re.findall(r'\d+', num)[0]

        # look only at text before match
        before_text = text[:pos]

        # Find closest measure before number - case sensitive
        nearest_measure, nearest_measure_pos = None, -1
        for measure in measure_keywords:
            p = before_text.rfind(measure)
            if p > nearest_measure_pos:
                nearest_measure = measure
                nearest_measure_pos = p

        # Find closest vessel before number - case insensitive
        nearest_vessel, nearest_vessel_pos = None, -1
        for vessel_name, vessel_key in vessel_keywords.items():
            for vessel in vessel_key:
                p = before_text.lower().rfind(vessel.lower())
                if p > nearest_vessel_pos:
                    nearest_vessel = vessel_name
                    nearest_vessel_pos = p

        # Calculate distances (in words)
        measure_distance = word_distance(text, nearest_measure_pos, pos) if nearest_measure_pos != -1 else float('inf')
        vessel_distance = word_distance(text, nearest_vessel_pos, pos) if nearest_vessel_pos != -1 else float('inf')

        # Skip if both are too far
        if min(measure_distance, vessel_distance) > MAX_WORD_DISTANCE:
            continue

        # Create column name
        if nearest_vessel and vessel_distance <= MAX_WORD_DISTANCE:
            colname = f"{nearest_vessel.replace(' ', '_')}_{nearest_measure if nearest_measure else default_measure}"
        elif nearest_measure and measure_distance <= MAX_WORD_DISTANCE:
            colname = nearest_measure.upper()
        else:
            continue

        # Store multiple values as list
        if colname not in results:
            results[colname] = [num_val]
        else:
            results[colname].append(num_val)

    # Convert lists to ';'-separated strings
    for key in results:
        results[key] = '; '.join(results[key])

    return pd.Series(results)


def extract_complications(text):
    # if there is no text return 0 
    if not isinstance(text, str) or text.strip() == "":
        return pd.Series({"Complicacoes": 0})

    # Normalize accents and case
    text_norm = ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    ).lower()

    # Look for "complica" in text
    idx = text_norm.find("complica")
    if idx == -1:
        # No "complica" found
        value = 0
    else:
        # Check if "sem" appears immediately before (within 5 chars)
        # This handles "sem complicacao" and small spacing
        window_start = max(0, idx-5)
        window = text_norm[window_start:idx]
        if "sem" in window.split():
            value = 0
        elif "ausencia de" in window.split():
            value = 0
        else:
            value = 1

    return pd.Series({"Complicacoes": value})



def extract_sucesso(text):

    # Always return a Series
    if not isinstance(text, str) or text.strip() == "":
        return pd.Series({"Sucesso": 0})

    # Normalize accents and case
    text_norm = ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    ).lower()

    # Check for "bom resultado" (accent-insensitive)
    if "bom resultado" in text_norm:
        value = 1
    elif "sucesso" in text_norm:
        value = 1
    else:
        value = 0

    return pd.Series({"Sucesso": value})


# Read data
df = pd.read_csv('data/reports_groundtruth.csv',index_col=0)

# Create columun with clean text
df['clean_text'] = df['Conclusões'].fillna('').astype(str).apply(
    lambda x: ''.join(
        c for c in unicodedata.normalize('NFKD', x)
        if not unicodedata.combining(c)
    )
)

# create cols for each section of report
top_separators =  ['CORONARIOGRAFIA','VENTRICULOGRAFIA','ANGIOPLASTIA',
                       'CONCLUSAO','NOTA']
df = structure_text_column(df,'clean_text', top_separators)
# save processed data
# df.to_csv('data/reports_train_clean.csv')


# Name of predictions
col_names = ['Tronco_Comum','Descendente_Anterior','Circunflexa','Coronária_Direita','Outras_artérias']
# Names to look for in text
locations = ['descendente anterior','tronco comum', 'coronaria direita','circunflexa', 'obtusa marginal']
tipo = ['FFR', 'iFR']
df_extracted = df[['clean_text','angioplastia', 'conclusao']].copy()

# Define known vessel/region keywords and col names
vessel_keywords = {'Descendente Anterior': ['Descendente Anterior'],
    'Coronária Direita': ['Coronaria Direita', 'Descendente Posterior'],
    'Tronco Comum': ['Tronco Comum'],
    'Circunflexa': ['Circunflexa'],
    'Outras_artérias': ['Marginal Obtusa','Ramo Intermedio', 'Bypass']}

# Extract FFR/iFR
# Define physiological measure keywords
measure_keywords = ['iFR', 'FFR']
MAX_WORD_DISTANCE = 40 # Define maximum word distance threshold
default_measure = 'iFR'
extracted = df_extracted['clean_text'].apply(extract_vessel_measures, args=(measure_keywords, vessel_keywords, MAX_WORD_DISTANCE,default_measure))
df_extracted = pd.concat([df_extracted, extracted], axis=1)

# Extract "tipo"
df_extracted.loc[df_extracted['angioplastia'].isna(),'Tipo'] = 'Coronariografia'
df_extracted.loc[df_extracted['angioplastia'].notna(),'Tipo'] = 'Coronariografia e Angioplastia	'


# Extract stent measures
# Define stent keywords
measure_keywords = ['stent']
MAX_WORD_DISTANCE = 20 # Define maximum word distance threshold
default_measure = 'stent'
extracted = df_extracted['angioplastia'].apply(extract_stent_measures, args=(measure_keywords, vessel_keywords, MAX_WORD_DISTANCE,default_measure))
df_extracted = pd.concat([df_extracted, extracted], axis=1)

# Extract complications
extracted = df_extracted['clean_text'].apply(extract_complications)
df_extracted = pd.concat([df_extracted, extracted], axis=1)

# Extract sucesso
extracted = df_extracted['conclusao'].apply(extract_sucesso)
df_extracted = pd.concat([df_extracted, extracted], axis=1)


# Prepare for evaluation
df_extracted.rename(columns={'clean_text':'Conclusões',
                             'Tronco_Comum_stent': 'Comprimento_Stents_mm_Tronco_Comum',
                            'Descendente_Anterior_stent': 'Comprimento_Stents_mm_Descendente_Anterior',
                            'Circunflexa_stent': 'Comprimento_Stents_mm_Circunflexa',
                            'Coronária_Direita_stent': 'Comprimento_Stents_mm_Coronária_Direita',
                            'Outras_artérias_stent': 'Comprimento_Stents_mm_Outras_artérias',

                             },inplace=True)
# columns_to_drop = ['angioplastia', 'conclusao']
# for col in columns_to_drop:
#     if col in df_extracted.columns:
#         df_extracted.drop(columns=col,inplace=True)

#Maybe no stents here? cretae column anyway with Nan
if 'Comprimento_Stents_mm_Outras_artérias' not in df_extracted.columns:
    df_extracted['Comprimento_Stents_mm_Outras_artérias'] = float('nan')


# List of stent measurement columns and the corresponding count columns
stent_cols = [
    'Comprimento_Stents_mm_Tronco_Comum',
    'Comprimento_Stents_mm_Circunflexa',
    'Comprimento_Stents_mm_Coronária_Direita',
    'Comprimento_Stents_mm_Descendente_Anterior',
    'Comprimento_Stents_mm_Outras_artérias'
]

count_cols = [
    'Nr_Stents_Tronco_Comum',
    'Nr_Stents_Circunflexa',
    'Nr_Stents_Coronária_Direita',
    'Nr_Stents_Descendente_Anterior',
    'Nr_Stents_Outras_artérias'
]

# Loop over each column
for stent_col, count_col in zip(stent_cols, count_cols):
    # If the column does not exist, create it filled with NaN
    if stent_col not in df_extracted.columns:
        df_extracted[stent_col] = float('nan')
    
    # Count the number of stents (split by ';'), handle NaN
    df_extracted[count_col] = df_extracted[stent_col].apply(
        lambda x: len(str(x).split(';')) if pd.notna(x) and str(x).strip() != '' else 0
    )

final_variables = ["Conclusões",
                    "Tipo",
                    'Tronco_Comum_FFR', 'Descendente_Anterior_FFR', 'Circunflexa_FFR', 'Coronária_Direita_FFR', 'Outras_artérias_FFR',
                    'Tronco_Comum_iFR', 'Descendente_Anterior_iFR', 'Circunflexa_iFR', 'Coronária_Direita_iFR', 'Outras_artérias_iFR',
                    'Complicacoes',
                    'Sucesso'] + stent_cols + count_cols

df_extracted = df_extracted[final_variables]

# Save extraction
os.makedirs("results/ie_regex", exist_ok=True)
df_extracted.to_csv('results/ie_regex/extraction_results.csv')
df_extracted.to_excel('results/ie_regex/extraction_results.xlsx')
