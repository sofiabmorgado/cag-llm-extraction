"""Extraction with baseline LLMs."""

import json
import pandas as pd
from guidance import assistant, models, gen, system, user
import guidance
import os, sys

def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, rel_path)


def clean_value(v):
    if pd.isna(v):
        return ""
    try:
        return v.item() if hasattr(v, 'item') else v
    except Exception:
        return v

def load_examples():
    train_df = pd.read_csv(resource_path("data/examples.csv")) 
    fields = [c for c in train_df.columns if c != "Conclusões"]
    examples = []
    for _, row in train_df.iterrows():
        gt = {f: clean_value(row[f]) for f in fields}
        user_prompt = "Extrai os seguintes campos" + ", ".join(fields) + " do seguinte relatório: " + str(row.get("Conclusões", ""))
        examples.append({"prompt": user_prompt, "reply": json.dumps(gt, ensure_ascii=False)})
    return examples


def build_lm(schema):
    global base_lm
    if base_lm is not None:
        return base_lm.copy()

    with user(): #system for all the models except mistral, user
        descriptions = []

        for field, props in schema["properties"].items():
            desc = props.get("description", "")
            if "enum" in props:
                enum_vals = ", ".join(f'"{v}"' for v in props["enum"])
                desc = f"{desc} (possible values: {enum_vals})"
            descriptions.append(f"{field}: {desc}")

        base_lm = (
            model
            + "És um especialista em relatórios de angiografias/coronariografia e angioplastia. O relatório inclui informação relativa aos indices de fisiologia: fractional flow reserve ou FFR e instant wave-free ratio ou iFR."
            "Dá como output um JSON válido. Usa estas descrições como referência:\n\n"
            + "\n".join(descriptions)
        )

    for ex in load_examples():
        with user():
            base_lm = base_lm + ex["prompt"]
        with assistant():
            base_lm = base_lm + ex["reply"]

    print("Few-shot model initialized.")
    return base_lm.copy()


def extract(input):
    
    schema_path = resource_path("output_schema_copied.json")
    schema = json.load(open(schema_path, "r"))

    lm = build_lm(schema)

    with user():
        lm += input
    with assistant():
        lm += guidance.json(schema=schema, name="res")

    return lm["res"]


base_path = resource_path('models')
model_path = os.path.join(base_path, 'gpt-oss-20b-F16.gguf')
model = models.LlamaCpp(model_path, n_gpu_layers=-1, n_ctx=4096)
base_lm = None