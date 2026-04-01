# Databricks notebook source

# MAGIC %md
# MAGIC # 02 - Create Clean Dataset
# MAGIC
# MAGIC **Objective:** Build a reproducible cleaned dataset from the raw clinical notes source.
# MAGIC
# MAGIC **Input:** raw CSV
# MAGIC
# MAGIC **Output:** `data/clean/clinical_notes_clean.parquet`

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install pyyaml --quiet

# COMMAND ----------

# DBTITLE 1,Imports
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Load the YAML project config."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clean_dataset(
    df: pd.DataFrame,
    min_class_samples: int,
    min_doc_length: int,
) -> pd.DataFrame:
    """Normalize and filter the raw dataset into a clean dataset."""
    normalized = df.copy()
    normalized.columns = (
        normalized.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
    )

    required_columns = {"transcription", "medical_specialty"}
    missing_columns = required_columns - set(normalized.columns)
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_columns)}")

    clean_df = normalized[["transcription", "medical_specialty"]].copy()
    clean_df = clean_df.dropna(subset=["transcription", "medical_specialty"])
    clean_df["transcription"] = clean_df["transcription"].astype(str).str.strip()
    clean_df["medical_specialty"] = clean_df["medical_specialty"].astype(str).str.strip()

    clean_df = clean_df[clean_df["transcription"].str.len() > 0]
    clean_df = clean_df.drop_duplicates(subset=["transcription"]).reset_index(drop=True)

    clean_df["word_count"] = clean_df["transcription"].str.split().str.len()
    clean_df = clean_df[clean_df["word_count"] >= min_doc_length].copy()

    class_counts = clean_df["medical_specialty"].value_counts()
    keep_labels = class_counts[class_counts >= min_class_samples].index
    clean_df = clean_df[clean_df["medical_specialty"].isin(keep_labels)].copy()

    clean_df["char_len"] = clean_df["transcription"].str.len()
    clean_df = clean_df.sort_values(["medical_specialty", "word_count"], ascending=[True, False])
    return clean_df.reset_index(drop=True)


def show(obj):
    """Display helper that works in Databricks and plain Python."""
    if "display" in globals():
        display(obj)
    else:
        print(obj)


# COMMAND ----------

# DBTITLE 1,Parameters
try:
    dbutils.widgets.text("config_path", "configs/config.yaml")
    dbutils.widgets.text("input_path", "")
    dbutils.widgets.text("output_path", "")
    CONFIG_PATH = dbutils.widgets.get("config_path").strip() or "configs/config.yaml"
    INPUT_PATH = dbutils.widgets.get("input_path").strip()
    OUTPUT_PATH = dbutils.widgets.get("output_path").strip()
except NameError:
    CONFIG_PATH = "configs/config.yaml"
    INPUT_PATH = ""
    OUTPUT_PATH = ""

config = load_config(CONFIG_PATH)

if not INPUT_PATH:
    default_raw_file = Path(config["data"]["raw_dir"]) / "mtsamples.csv"
    INPUT_PATH = str(default_raw_file)

if not OUTPUT_PATH:
    OUTPUT_PATH = config["data"]["clean_file"]

print(f"Using CONFIG_PATH={CONFIG_PATH}")
print(f"Using INPUT_PATH={INPUT_PATH}")
print(f"Using OUTPUT_PATH={OUTPUT_PATH}")

# COMMAND ----------

# DBTITLE 1,Load raw dataset
raw_df = pd.read_csv(INPUT_PATH)
print(f"Raw shape: {raw_df.shape}")
print(f"Raw columns: {list(raw_df.columns)}")

# COMMAND ----------

# DBTITLE 1,Create clean dataset
clean_df = clean_dataset(
    raw_df,
    min_class_samples=config["data"]["min_class_samples"],
    min_doc_length=config["preprocessing"]["min_doc_length"],
)

print(f"Clean shape: {clean_df.shape}")
print(f"Unique specialties: {clean_df['medical_specialty'].nunique()}")
show(clean_df.head(10))

# COMMAND ----------

# DBTITLE 1,Write clean parquet
output_path = Path(OUTPUT_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)
clean_df.to_parquet(output_path, index=False)

print(f"Clean dataset written to: {output_path}")
print(f"Rows written: {len(clean_df)}")

# COMMAND ----------

# DBTITLE 1,Validation summary
summary = {
    "rows": len(clean_df),
    "specialties": clean_df["medical_specialty"].nunique(),
    "min_word_count": int(clean_df["word_count"].min()),
    "max_word_count": int(clean_df["word_count"].max()),
}
show(pd.DataFrame([summary]))
