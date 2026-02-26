"""For few-shot Implausible prompting results, run this code to remove the implausible values"""

import pandas as pd
import numpy as np

path = 'results/<model_name_folder>/extraction_results.xlsx'
df_pred = pd.read_csv(path)

df_numeric = df_pred.apply(pd.to_numeric, errors='coerce')

# Keep only values in [0,1], else NaN
df = df_numeric.where((df_numeric >= 0) & (df_numeric <= 1.1))
df["Conclusões"] = df_pred["Conclusões"].values

df.to_excel(path, index=False)