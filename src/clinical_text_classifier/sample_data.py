"""Generate synthetic sample data for pipeline validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SAMPLES = [
    (
        "The patient is a 45-year-old male with right knee pain after a fall. MRI shows a torn anterior cruciate ligament and arthroscopic reconstruction is recommended.",
        "Orthopedic",
    ),
    (
        "A 68-year-old male presents with chest pain and shortness of breath. EKG shows inferior ST elevation and emergent catheterization is planned.",
        "Cardiovascular / Pulmonary",
    ),
    (
        "A 52-year-old male presents with epigastric pain and melena. Endoscopy reveals a duodenal ulcer and proton pump inhibitor therapy is started.",
        "Gastroenterology",
    ),
    (
        "A 73-year-old female presents with acute right-sided weakness and aphasia. Imaging is concerning for left MCA occlusion and stroke treatment is initiated.",
        "Neurology",
    ),
    (
        "A 58-year-old male presents for annual follow-up with hypertension, diabetes, and hyperlipidemia requiring medication adjustment.",
        "General Medicine",
    ),
]


def generate_sample_dataset(output_path: str = "data/raw/mtsamples.csv") -> pd.DataFrame:
    """Generate a small MTSamples-like dataset for smoke testing."""
    df = pd.DataFrame(SAMPLES, columns=["transcription", "medical_specialty"])
    df.insert(0, "Unnamed: 0", range(len(df)))
    df.insert(1, "description", "Sample transcription")
    df.insert(3, "sample_name", "Sample " + df.index.astype(str))
    df.insert(4, "keywords", "")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df
