# Databricks notebook source

# MAGIC %md
# MAGIC # 03 - Create Train-Ready Dataset
# MAGIC
# MAGIC **Objective:** Build a reproducible train-ready dataset from the cleaned clinical notes dataset.
# MAGIC
# MAGIC **Input:** `data/clean/clinical_notes_clean.parquet`
# MAGIC
# MAGIC **Output:** `data/train_ready/clinical_text_train_ready.parquet`

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install pyyaml --quiet

# COMMAND ----------

# DBTITLE 1,Imports
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    """Load the YAML project config."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def add_dataset_splits(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> pd.DataFrame:
    """Add deterministic train/val/test splits with stratification by label."""
    if df.empty:
        raise ValueError("Clean dataset is empty; cannot create a train-ready dataset.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    labels = df["label"]

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_pool = df.loc[train_idx]

    if val_size > 0:
        val_fraction_of_train_pool = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_pool.index,
            test_size=val_fraction_of_train_pool,
            random_state=random_state,
            stratify=train_pool["label"],
        )
    else:
        val_idx = pd.Index([])

    split_df = df.copy()
    split_df["split"] = "train"
    split_df.loc[test_idx, "split"] = "test"
    if len(val_idx) > 0:
        split_df.loc[val_idx, "split"] = "val"

    return split_df


def create_train_ready_dataset(
    clean_df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> pd.DataFrame:
    """Create the final model-ready dataset schema."""
    required_columns = {"transcription", "medical_specialty", "word_count", "char_len"}
    missing_columns = required_columns - set(clean_df.columns)
    if missing_columns:
        raise ValueError(f"Clean dataset missing required columns: {sorted(missing_columns)}")

    train_ready_df = clean_df.copy()
    train_ready_df = train_ready_df.rename(
        columns={
            "transcription": "text",
            "medical_specialty": "label",
        }
    )

    train_ready_df["text"] = train_ready_df["text"].astype(str).str.strip()
    train_ready_df["label"] = train_ready_df["label"].astype(str).str.strip()
    train_ready_df = train_ready_df[(train_ready_df["text"].str.len() > 0) & (train_ready_df["label"].str.len() > 0)]
    train_ready_df = train_ready_df.reset_index(drop=True)

    train_ready_df = add_dataset_splits(
        train_ready_df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    split_order = {"train": 0, "val": 1, "test": 2}
    train_ready_df["split_order"] = train_ready_df["split"].map(split_order)
    train_ready_df = train_ready_df.sort_values(["split_order", "label", "word_count"])
    train_ready_df = train_ready_df.drop(columns=["split_order"]).reset_index(drop=True)

    ordered_columns = ["text", "label", "split", "word_count", "char_len"]
    return train_ready_df[ordered_columns]


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
    INPUT_PATH = config["data"]["clean_file"]

if not OUTPUT_PATH:
    OUTPUT_PATH = config["data"]["train_ready_file"]

print(f"Using CONFIG_PATH={CONFIG_PATH}")
print(f"Using INPUT_PATH={INPUT_PATH}")
print(f"Using OUTPUT_PATH={OUTPUT_PATH}")

# COMMAND ----------

# DBTITLE 1,Load clean dataset
clean_df = pd.read_parquet(INPUT_PATH)
print(f"Clean shape: {clean_df.shape}")
print(f"Clean columns: {list(clean_df.columns)}")

# COMMAND ----------

# DBTITLE 1,Create train-ready dataset
train_ready_df = create_train_ready_dataset(
    clean_df,
    test_size=config["data"]["test_size"],
    val_size=config["data"]["val_size"],
    random_state=config["project"]["seed"],
)

print(f"Train-ready shape: {train_ready_df.shape}")
show(train_ready_df.head(10))

# COMMAND ----------

# DBTITLE 1,Write train-ready parquet
output_path = Path(OUTPUT_PATH)
output_path.parent.mkdir(parents=True, exist_ok=True)
train_ready_df.to_parquet(output_path, index=False)

print(f"Train-ready dataset written to: {output_path}")
print(f"Rows written: {len(train_ready_df)}")

# COMMAND ----------

# DBTITLE 1,Validation summary
split_summary = (
    train_ready_df.groupby(["split", "label"])
    .size()
    .reset_index(name="rows")
    .sort_values(["split", "rows"], ascending=[True, False])
)

summary = {
    "rows": len(train_ready_df),
    "labels": train_ready_df["label"].nunique(),
    "train_rows": int((train_ready_df["split"] == "train").sum()),
    "val_rows": int((train_ready_df["split"] == "val").sum()),
    "test_rows": int((train_ready_df["split"] == "test").sum()),
}

show(pd.DataFrame([summary]))
show(split_summary.head(15))
