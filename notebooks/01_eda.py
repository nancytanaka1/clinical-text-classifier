# Databricks notebook source

# MAGIC %md
# MAGIC # 01 — Exploratory Data Analysis: Clinical Text Classifier
# MAGIC **Dataset:** MTSamples (4,542 medical transcription samples, 21 specialties)
# MAGIC
# MAGIC **Objective:** Characterize the dataset to inform preprocessing, feature engineering, and model selection.
# MAGIC
# MAGIC **Branch:** `feature/eda` → PR to `dev`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 | Environment Setup

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install textstat wordcloud scikit-learn matplotlib seaborn --quiet

# COMMAND ----------

# DBTITLE 1,Imports
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Tuple

# NLP
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import textstat
from wordcloud import WordCloud

# Spark (available natively in Databricks)
from pyspark.sql import functions as F

# Plot config
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
sns.set_theme(style="whitegrid", palette="viridis")

print("Environment ready.")


def show(obj):
    """Display helper that works in Databricks and plain Python."""
    if "display" in globals():
        display(obj)
    else:
        print(obj)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 | Load Data
# MAGIC
# MAGIC **Update `DATA_PATH` below** to point to your MTSamples CSV location.
# MAGIC Options:
# MAGIC - Databricks Repos: `/Workspace/Repos/<user>/clinical-text-classifier/data/raw/mtsamples.csv`
# MAGIC - DBFS upload: `/dbfs/FileStore/tables/mtsamples.csv`
# MAGIC - Unity Catalog Volume: `/Volumes/<catalog>/<schema>/<volume>/mtsamples.csv`

# COMMAND ----------

# DBTITLE 1,Configuration — UPDATE THIS PATH
DEFAULT_DATA_PATHS = [
    "/dbfs/FileStore/tables/mtsamples.csv",
    "/Workspace/Repos/nguyenn.mail@gmail.com/clinical-text-classifier/data/raw/mtsamples.csv",
    "/Workspace/Repos/nancytanaka1/clinical-text-classifier/data/raw/mtsamples.csv",
]

try:
    dbutils.widgets.text("data_path", "")
    widget_data_path = dbutils.widgets.get("data_path").strip()
except NameError:
    widget_data_path = ""

if widget_data_path:
    DATA_PATH = widget_data_path
else:
    existing_default = next((path for path in DEFAULT_DATA_PATHS if Path(path).exists()), None)
    DATA_PATH = existing_default or DEFAULT_DATA_PATHS[0]

print(f"Using DATA_PATH={DATA_PATH}")

# COMMAND ----------

# DBTITLE 1,Load and initial inspection
df = pd.read_csv(DATA_PATH)

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Key columns expected: medical_specialty, transcription, description, keywords
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head(3)

# COMMAND ----------

# DBTITLE 1,Data quality check
print("=== Missing Values ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
quality = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
quality = quality[quality.missing_count > 0].sort_values("missing_pct", ascending=False)
show(quality)

print(f"\n=== Duplicates ===")
n_dupes = df.duplicated(subset=["transcription"]).sum()
print(f"Duplicate transcriptions: {n_dupes} ({n_dupes/len(df)*100:.1f}%)")

# Drop rows with no transcription text — can't classify empty docs
df = df.dropna(subset=["transcription"]).reset_index(drop=True)
print(f"\nUsable rows after dropping null transcriptions: {len(df)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 | Label Distribution & Class Imbalance

# COMMAND ----------

# DBTITLE 1,Specialty frequency table
specialty_counts = (
    df["medical_specialty"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "specialty", "medical_specialty": "specialty", "count": "n"})
)
# Handle different pandas versions
if "n" not in specialty_counts.columns:
    specialty_counts.columns = ["specialty", "n"]

specialty_counts["pct"] = (specialty_counts["n"] / specialty_counts["n"].sum() * 100).round(2)
specialty_counts["cumulative_pct"] = specialty_counts["pct"].cumsum().round(2)

show(specialty_counts)

# COMMAND ----------

# DBTITLE 1,Label distribution — bar chart
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Bar chart
order = df["medical_specialty"].value_counts().index
sns.countplot(y="medical_specialty", data=df, order=order, ax=axes[0], palette="viridis")
axes[0].set_title("Sample Count by Medical Specialty")
axes[0].set_xlabel("Count")
axes[0].set_ylabel("")

# Cumulative % (Pareto)
specialty_sorted = df["medical_specialty"].value_counts()
cumulative = specialty_sorted.cumsum() / specialty_sorted.sum() * 100
axes[1].bar(range(len(specialty_sorted)), specialty_sorted.values, color="steelblue", alpha=0.7)
axes[1].set_ylabel("Count", color="steelblue")
ax2 = axes[1].twinx()
ax2.plot(range(len(specialty_sorted)), cumulative.values, color="red", marker="o", linewidth=2)
ax2.set_ylabel("Cumulative %", color="red")
ax2.axhline(y=80, color="red", linestyle="--", alpha=0.5, label="80% threshold")
axes[1].set_title("Pareto Chart — Specialty Distribution")
axes[1].set_xticks(range(len(specialty_sorted)))
axes[1].set_xticklabels(specialty_sorted.index, rotation=75, ha="right", fontsize=9)
ax2.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Class imbalance metrics
n_classes = df["medical_specialty"].nunique()
majority_class = specialty_counts.iloc[0]
minority_class = specialty_counts.iloc[-1]
imbalance_ratio = majority_class["n"] / minority_class["n"]

print(f"Number of classes:    {n_classes}")
print(f"Majority class:       {majority_class['specialty']} (n={majority_class['n']}, {majority_class['pct']}%)")
print(f"Minority class:       {minority_class['specialty']} (n={minority_class['n']}, {minority_class['pct']}%)")
print(f"Imbalance ratio:      {imbalance_ratio:.1f}:1")
print(f"Effective n per class: {len(df) / n_classes:.0f} (if balanced)")

# Flag classes with < 50 samples — likely need to merge or drop
thin_classes = specialty_counts[specialty_counts["n"] < 50]
print(f"\nClasses with < 50 samples ({len(thin_classes)}):")
for _, row in thin_classes.iterrows():
    print(f"  - {row['specialty']}: {row['n']} samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imbalance Strategy Recommendations
# MAGIC
# MAGIC | Imbalance Ratio | Strategy |
# MAGIC |---|---|
# MAGIC | < 5:1 | Standard training, monitor per-class metrics |
# MAGIC | 5–20:1 | Class weights, stratified splits, focal loss |
# MAGIC | > 20:1 | SMOTE/oversampling, class merging, or hierarchical classification |
# MAGIC
# MAGIC **Action items** will be documented in the summary cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 | Text Length & Token Analysis

# COMMAND ----------

# DBTITLE 1,Compute text features
def compute_text_stats(text: str) -> Dict[str, float]:
    """Compute basic text statistics for a single document."""
    if not isinstance(text, str):
        return {"char_len": 0, "word_count": 0, "sentence_count": 0, "avg_word_len": 0}
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    return {
        "char_len": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_len": round(avg_word_len, 2),
    }


stats = df["transcription"].apply(compute_text_stats).apply(pd.Series)
df = pd.concat([df, stats], axis=1)

print("=== Text Length Statistics ===")
show(df[["char_len", "word_count", "sentence_count", "avg_word_len"]].describe().round(1))

# COMMAND ----------

# DBTITLE 1,Text length distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col, title in zip(
    axes,
    ["word_count", "char_len", "sentence_count"],
    ["Word Count", "Character Length", "Sentence Count"],
):
    sns.histplot(df[col], bins=50, ax=ax, kde=True, color="steelblue")
    ax.axvline(df[col].median(), color="red", linestyle="--", label=f"Median: {df[col].median():.0f}")
    ax.set_title(f"Distribution of {title}")
    ax.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Text length by specialty — boxplots
fig, ax = plt.subplots(figsize=(16, 8))
order = df.groupby("medical_specialty")["word_count"].median().sort_values(ascending=False).index
sns.boxplot(x="word_count", y="medical_specialty", data=df, order=order, ax=ax, palette="viridis")
ax.set_title("Word Count Distribution by Specialty")
ax.set_xlabel("Word Count")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Flag outliers
q1 = df["word_count"].quantile(0.25)
q3 = df["word_count"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df["word_count"] < lower_bound) | (df["word_count"] > upper_bound)]
print(f"Word count IQR: [{q1:.0f}, {q3:.0f}], bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")
print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
print(f"  Very short (<{lower_bound:.0f} words): {len(df[df['word_count'] < lower_bound])}")
print(f"  Very long  (>{upper_bound:.0f} words): {len(df[df['word_count'] > upper_bound])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 | TF-IDF — Top Terms per Specialty

# COMMAND ----------

# DBTITLE 1,TF-IDF extraction
def get_top_tfidf_terms(
    df: pd.DataFrame,
    text_col: str = "transcription",
    label_col: str = "medical_specialty",
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract top TF-IDF terms per class label."""
    tfidf = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 1),
        min_df=3,
        max_df=0.85,
    )
    text_series = df[text_col].fillna("").astype(str)
    label_series = df[label_col]

    tfidf_matrix = tfidf.fit_transform(text_series)
    feature_names = tfidf.get_feature_names_out()

    results = []
    for label in label_series.dropna().unique():
        mask = (label_series == label).to_numpy()
        if mask.sum() == 0:
            continue

        class_tfidf = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
        top_indices = class_tfidf.argsort()[-top_n:][::-1]
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "specialty": label,
                "rank": rank,
                "term": feature_names[idx],
                "tfidf_score": round(float(class_tfidf[idx]), 4),
            })
    return pd.DataFrame(results)


tfidf_top = get_top_tfidf_terms(df)
show(tfidf_top.head(30))

# COMMAND ----------

# DBTITLE 1,Heatmap — Top 10 TF-IDF terms for top 10 specialties
top_10_specialties = df["medical_specialty"].value_counts().head(10).index.tolist()
subset = tfidf_top[
    (tfidf_top["specialty"].isin(top_10_specialties)) & (tfidf_top["rank"] <= 10)
]
pivot = subset.pivot_table(index="specialty", columns="term", values="tfidf_score", fill_value=0)

fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(pivot, cmap="YlOrRd", annot=False, ax=ax, linewidths=0.5)
ax.set_title("TF-IDF Heatmap — Top 10 Terms × Top 10 Specialties")
plt.xticks(rotation=60, ha="right", fontsize=9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 | N-gram Analysis (Bigrams & Trigrams)

# COMMAND ----------

# DBTITLE 1,Extract top n-grams per specialty
def get_top_ngrams(
    corpus: pd.Series,
    ngram_range: Tuple[int, int] = (2, 2),
    top_n: int = 15,
) -> List[Tuple[str, int]]:
    """Extract top n-grams from a text corpus."""
    vec = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    bow = vec.fit_transform(corpus.fillna(""))
    sum_words = bow.sum(axis=0).A1
    words_freq = [(word, int(sum_words[idx])) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_n]


# Global bigrams and trigrams
top_bigrams = get_top_ngrams(df["transcription"], ngram_range=(2, 2), top_n=20)
top_trigrams = get_top_ngrams(df["transcription"], ngram_range=(3, 3), top_n=20)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Bigrams
bg_df = pd.DataFrame(top_bigrams, columns=["bigram", "count"])
sns.barplot(x="count", y="bigram", data=bg_df, ax=axes[0], palette="Blues_r")
axes[0].set_title("Top 20 Bigrams (All Specialties)")

# Trigrams
tg_df = pd.DataFrame(top_trigrams, columns=["trigram", "count"])
sns.barplot(x="count", y="trigram", data=tg_df, ax=axes[1], palette="Oranges_r")
axes[1].set_title("Top 20 Trigrams (All Specialties)")

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Per-specialty bigrams (top 5 specialties)
top_5_specialties = df["medical_specialty"].value_counts().head(5).index.tolist()

fig, axes = plt.subplots(1, 5, figsize=(28, 6), sharey=False)
for ax, spec in zip(axes, top_5_specialties):
    subset = df[df["medical_specialty"] == spec]
    bigrams = get_top_ngrams(subset["transcription"], ngram_range=(2, 2), top_n=10)
    bg = pd.DataFrame(bigrams, columns=["bigram", "count"])
    sns.barplot(x="count", y="bigram", data=bg, ax=ax, palette="viridis")
    ax.set_title(spec, fontsize=10)
    ax.set_ylabel("")
    ax.set_xlabel("")

plt.suptitle("Top 10 Bigrams per Specialty (Top 5 Classes)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 | Vocabulary Overlap Analysis

# COMMAND ----------

# DBTITLE 1,Jaccard similarity between specialty vocabularies
def get_vocabulary(texts: pd.Series, min_freq: int = 3) -> set:
    """Get unique vocabulary from a text corpus."""
    vec = CountVectorizer(stop_words="english", min_df=min_freq)
    vec.fit(texts.fillna(""))
    return set(vec.get_feature_names_out())


specialties = df["medical_specialty"].value_counts().head(10).index.tolist()
vocabs = {spec: get_vocabulary(df[df["medical_specialty"] == spec]["transcription"]) for spec in specialties}

# Jaccard similarity matrix
jaccard_matrix = pd.DataFrame(index=specialties, columns=specialties, dtype=float)
for s1 in specialties:
    for s2 in specialties:
        intersection = len(vocabs[s1] & vocabs[s2])
        union = len(vocabs[s1] | vocabs[s2])
        jaccard_matrix.loc[s1, s2] = round(intersection / union, 3) if union > 0 else 0

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    jaccard_matrix.astype(float),
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    ax=ax,
    linewidths=0.5,
    vmin=0,
    vmax=1,
)
ax.set_title("Jaccard Vocabulary Similarity — Top 10 Specialties")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Unique terms per specialty (discriminative vocabulary)
all_vocab = set()
for v in vocabs.values():
    all_vocab |= v

print("=== Vocabulary Stats ===")
print(f"Total unique terms (top 10 specialties): {len(all_vocab)}")
print()
for spec in specialties:
    other_vocabs = set()
    for s, v in vocabs.items():
        if s != spec:
            other_vocabs |= v
    unique_to_spec = vocabs[spec] - other_vocabs
    print(f"{spec}: {len(vocabs[spec])} total, {len(unique_to_spec)} unique ({len(unique_to_spec)/len(vocabs[spec])*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 | Readability Scores

# COMMAND ----------

# DBTITLE 1,Compute readability metrics
def compute_readability(text: str) -> Dict[str, float]:
    """Compute readability scores using textstat."""
    if not isinstance(text, str) or len(text.split()) < 10:
        return {
            "flesch_reading_ease": np.nan,
            "flesch_kincaid_grade": np.nan,
            "gunning_fog": np.nan,
            "coleman_liau": np.nan,
        }
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "coleman_liau": textstat.coleman_liau_index(text),
    }


readability = df["transcription"].apply(compute_readability).apply(pd.Series)
df = pd.concat([df, readability], axis=1)

print("=== Readability Score Summary ===")
show(df[["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog", "coleman_liau"]].describe().round(2))

# COMMAND ----------

# DBTITLE 1,Readability by specialty
read_by_spec = (
    df.groupby("medical_specialty")[["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"]]
    .median()
    .sort_values("flesch_reading_ease")
)

fig, ax = plt.subplots(figsize=(14, 8))
read_by_spec.plot(kind="barh", ax=ax)
ax.set_title("Median Readability Scores by Specialty")
ax.set_xlabel("Score")
ax.set_ylabel("")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 | Word Clouds

# COMMAND ----------

# DBTITLE 1,Word clouds — top 6 specialties
top_6 = df["medical_specialty"].value_counts().head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for ax, spec in zip(axes.flatten(), top_6):
    text = " ".join(df[df["medical_specialty"] == spec]["transcription"].fillna(""))
    wc = WordCloud(
        width=600,
        height=400,
        background_color="white",
        max_words=80,
        colormap="viridis",
        stopwords=set(WordCloud().stopwords) | {"patient", "procedure", "performed", "using", "placed"},
    ).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(spec, fontsize=12, fontweight="bold")
    ax.axis("off")

plt.suptitle("Word Clouds by Specialty (Top 6 Classes)", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 | Summary & Actionable Findings

# COMMAND ----------

# DBTITLE 1,Automated summary
print("=" * 70)
print("EDA SUMMARY — Clinical Text Classifier")
print("=" * 70)

n_total = len(df)
n_classes = df["medical_specialty"].nunique()
majority = df["medical_specialty"].value_counts().iloc[0]
minority = df["medical_specialty"].value_counts().iloc[-1]
ratio = majority / minority

print(f"""
DATASET
  Total samples:        {n_total}
  Unique specialties:   {n_classes}
  Imbalance ratio:      {ratio:.1f}:1 (majority/minority)

TEXT CHARACTERISTICS
  Median word count:    {df['word_count'].median():.0f}
  Mean word count:      {df['word_count'].mean():.0f}
  Std word count:       {df['word_count'].std():.0f}
  Median Flesch-Kincaid grade: {df['flesch_kincaid_grade'].median():.1f}

CLASS IMBALANCE
  Classes with < 50 samples: {len(specialty_counts[specialty_counts['n'] < 50])}
  Classes with < 100 samples: {len(specialty_counts[specialty_counts['n'] < 100])}
  Top 5 classes hold: {specialty_counts.head(5)['pct'].sum():.1f}% of data

RECOMMENDED ACTIONS
  1. MERGE or DROP thin classes (<50 samples) — underpowered for supervised learning
  2. APPLY class weights or focal loss — imbalance ratio >{ratio:.0f}:1 demands it
  3. STRATIFIED splits mandatory — random splits will under-represent minority classes
  4. TRUNCATE/PAD texts — high variance in doc length; consider max_length cutoff
  5. REMOVE BOILERPLATE — shared medical phrases inflate vocabulary overlap
  6. CONSIDER HIERARCHICAL classification if merging reduces to <10 macro-categories
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC | Step | Task | Priority |
# MAGIC |------|------|----------|
# MAGIC | 1 | Define class merging strategy based on thin classes + Jaccard overlap | **HIGH** |
# MAGIC | 2 | Text preprocessing pipeline (lowercase, stopwords, lemmatization, boilerplate removal) | **HIGH** |
# MAGIC | 3 | Train/val/test stratified split (70/15/15) | **HIGH** |
# MAGIC | 4 | Baseline model: TF-IDF + Logistic Regression (macro F1) | MEDIUM |
# MAGIC | 5 | Fine-tune ClinicalBERT / BioBERT on the classification task | MEDIUM |
# MAGIC | 6 | MLflow experiment tracking setup | MEDIUM |
