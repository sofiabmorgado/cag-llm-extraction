"""Extraction with constrained LLMs."""

import json
import pandas as pd
from tqdm import tqdm
import backbone_extractor_constrained_llms as extractor

def safe_extract(x):
    try:
        return extractor.extract(str(x))
    except Exception:
        return None

#RUN
output_path = "results/<model_name_folder>/extraction.csv"
checkpoint = 10

schema_path = extractor.resource_path("output_schema.json")
schema = json.load(open(schema_path, "r"))

df = pd.read_csv("data/reports_groundtruth.csv")
df["extracted"] = None

extractor.build_lm(schema)

for i, x in enumerate(tqdm(df["Conclusões"], total=len(df))):
    df.at[i, "extracted"] = safe_extract(x)

    if (i + 1) % checkpoint == 0 or (i + 1) == len(df):
        df.to_csv(output_path, index=False)
