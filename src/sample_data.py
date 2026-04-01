"""
Generate a synthetic sample dataset mirroring MTSamples schema.

Use this for pipeline testing when the real dataset isn't available.
Replace with real MTSamples data for actual model training.
"""

from pathlib import Path

import pandas as pd

# Realistic sample transcriptions by specialty (abbreviated for testing)
SAMPLES = [
    # Orthopedic
    (
        "The patient is a 45-year-old male who presents with right knee pain and "
        "swelling after a fall. MRI reveals a torn anterior cruciate ligament. "
        "Physical examination demonstrates positive Lachman test and anterior "
        "drawer sign. Recommendation is for arthroscopic ACL reconstruction.",
        "Orthopedic",
    ),
    (
        "Patient presents with chronic low back pain radiating to the left lower "
        "extremity. Lumbar MRI shows L4-L5 disc herniation with nerve root "
        "compression. Conservative treatment with physical therapy and epidural "
        "steroid injections is recommended before considering surgical intervention.",
        "Orthopedic",
    ),
    (
        "This is a 62-year-old female with severe osteoarthritis of the left hip. "
        "X-rays demonstrate joint space narrowing, osteophyte formation, and "
        "subchondral sclerosis. Patient has failed conservative management. "
        "Total hip arthroplasty is recommended.",
        "Orthopedic",
    ),
    (
        "The patient is a 28-year-old male athlete with recurrent right shoulder "
        "dislocations. MRI arthrogram shows a Bankart lesion. Arthroscopic "
        "Bankart repair is planned. Pre-operative labs and EKG are within "
        "normal limits.",
        "Orthopedic",
    ),
    (
        "Patient is a 55-year-old female with bilateral carpal tunnel syndrome "
        "confirmed by nerve conduction studies. She reports numbness and tingling "
        "in both hands worse at night. Endoscopic carpal tunnel release is scheduled.",
        "Orthopedic",
    ),
    # Cardiovascular / Pulmonary
    (
        "The patient is a 68-year-old male with a history of hypertension, "
        "hyperlipidemia, and type 2 diabetes presenting with chest pain and "
        "shortness of breath. EKG shows ST elevation in leads II, III, and aVF. "
        "Troponin levels are elevated. Emergent cardiac catheterization is planned.",
        "Cardiovascular / Pulmonary",
    ),
    (
        "Patient is a 72-year-old female with congestive heart failure, "
        "ejection fraction of 30%. Echocardiogram demonstrates dilated left "
        "ventricle with global hypokinesis. BNP is markedly elevated. Medical "
        "management with ACE inhibitor, beta-blocker, and diuretic is optimized.",
        "Cardiovascular / Pulmonary",
    ),
    (
        "This is a 58-year-old male presenting with exertional dyspnea and "
        "bilateral lower extremity edema. Right heart catheterization reveals "
        "elevated pulmonary artery pressures. Diagnosed with pulmonary arterial "
        "hypertension. Started on sildenafil and ambrisentan.",
        "Cardiovascular / Pulmonary",
    ),
    (
        "Patient is a 65-year-old male with atrial fibrillation and rapid "
        "ventricular response. Heart rate is 142 beats per minute. IV diltiazem "
        "drip is initiated for rate control. TEE is negative for left atrial "
        "thrombus. Cardioversion is planned.",
        "Cardiovascular / Pulmonary",
    ),
    (
        "The patient is a 50-year-old female with severe aortic stenosis. "
        "Echocardiogram shows aortic valve area of 0.7 cm2 with a mean gradient "
        "of 48 mmHg. She is symptomatic with syncope and dyspnea. Referred for "
        "transcatheter aortic valve replacement.",
        "Cardiovascular / Pulmonary",
    ),
    # Gastroenterology
    (
        "Patient is a 52-year-old male presenting with epigastric pain, nausea, "
        "and melena. Upper endoscopy reveals a 2 cm duodenal ulcer with clean "
        "base. Biopsies are taken for H. pylori. Started on proton pump inhibitor "
        "therapy and triple therapy for eradication.",
        "Gastroenterology",
    ),
    (
        "This is a 47-year-old female with chronic diarrhea, abdominal cramping, "
        "and weight loss. Colonoscopy demonstrates patchy inflammation in the "
        "terminal ileum consistent with Crohn disease. Biopsies confirm "
        "granulomatous inflammation. Started on mesalamine.",
        "Gastroenterology",
    ),
    (
        "Patient is a 60-year-old male presenting for screening colonoscopy. "
        "A 15 mm sessile polyp is identified in the ascending colon and removed "
        "by endoscopic mucosal resection. Pathology reveals tubular adenoma with "
        "low-grade dysplasia. Follow-up colonoscopy in 3 years.",
        "Gastroenterology",
    ),
    (
        "The patient is a 38-year-old female with elevated liver enzymes. "
        "Hepatitis B and C serologies are negative. ANA is positive at 1:640. "
        "Liver biopsy demonstrates interface hepatitis consistent with autoimmune "
        "hepatitis. Started on prednisone and azathioprine.",
        "Gastroenterology",
    ),
    (
        "Patient is a 55-year-old male with cirrhosis secondary to alcohol use. "
        "Presenting with tense ascites and lower extremity edema. Large volume "
        "paracentesis performed removing 6 liters of transudative fluid. "
        "SBP ruled out. Diuretics adjusted.",
        "Gastroenterology",
    ),
    # Neurology
    (
        "The patient is a 73-year-old female presenting with acute onset "
        "right-sided weakness and aphasia. CT head is negative for hemorrhage. "
        "CT angiography shows left MCA occlusion. tPA is administered within "
        "the 4.5 hour window. Thrombectomy is considered.",
        "Neurology",
    ),
    (
        "Patient is a 35-year-old female with recurrent episodes of optic neuritis "
        "and numbness in the lower extremities. MRI brain shows multiple "
        "periventricular white matter lesions. CSF shows oligoclonal bands. "
        "Diagnosed with relapsing-remitting multiple sclerosis.",
        "Neurology",
    ),
    (
        "This is a 68-year-old male with progressive memory loss, word-finding "
        "difficulty, and personality changes over the past 2 years. MRI shows "
        "bilateral hippocampal atrophy. Neuropsychological testing confirms "
        "significant cognitive decline. Diagnosed with probable Alzheimer disease.",
        "Neurology",
    ),
    (
        "Patient is a 29-year-old female with recurrent headaches described as "
        "unilateral throbbing with photophobia, phonophobia, and nausea. "
        "Episodes last 4-72 hours. Frequency is 8 per month. Diagnosed with "
        "chronic migraine. Started on topiramate for prophylaxis.",
        "Neurology",
    ),
    (
        "The patient is a 62-year-old male with progressive resting tremor, "
        "bradykinesia, and rigidity predominantly affecting the right upper "
        "extremity. DaTscan shows reduced dopaminergic activity in the left "
        "putamen. Diagnosed with Parkinson disease. Started on carbidopa-levodopa.",
        "Neurology",
    ),
    # General Medicine
    (
        "Patient is a 58-year-old male presenting for annual physical examination. "
        "Medical history includes hypertension, type 2 diabetes, and "
        "hyperlipidemia. HbA1c is 7.8%. LDL is 142. Blood pressure is 148/92. "
        "Medications are adjusted and lifestyle modifications are reinforced.",
        "General Medicine",
    ),
    (
        "This is a 42-year-old female presenting with fatigue, weight gain, and "
        "cold intolerance. TSH is elevated at 12.4 with low free T4. Diagnosed "
        "with primary hypothyroidism. Started on levothyroxine 50 mcg daily "
        "with TSH recheck in 6 weeks.",
        "General Medicine",
    ),
    (
        "Patient is a 65-year-old male with poorly controlled type 2 diabetes on "
        "metformin and glipizide. HbA1c is 9.2%. Fasting glucose is 210. Added "
        "insulin glargine 10 units at bedtime. Dietary counseling and diabetic "
        "eye exam ordered.",
        "General Medicine",
    ),
    (
        "The patient is a 50-year-old female with recurrent urinary tract "
        "infections. Urinalysis shows pyuria and bacteriuria. Culture grows E. "
        "coli sensitive to nitrofurantoin. Treated with nitrofurantoin 100 mg "
        "twice daily for 5 days. Discussed prophylactic strategies.",
        "General Medicine",
    ),
    (
        "Patient is a 70-year-old male presenting with community-acquired "
        "pneumonia. Chest X-ray shows right lower lobe consolidation. Started "
        "on ceftriaxone and azithromycin. Oxygen saturation is 93% on room air. "
        "Admitted for inpatient management.",
        "General Medicine",
    ),
]


def generate_sample_dataset(output_path: str = "data/raw/mtsamples.csv") -> pd.DataFrame:
    """Generate a synthetic MTSamples-like dataset for pipeline testing.

    Args:
        output_path: Where to save the CSV.

    Returns:
        DataFrame with transcription and medical_specialty columns.
    """
    df = pd.DataFrame(SAMPLES, columns=["transcription", "medical_specialty"])

    # Add the extra columns that exist in real MTSamples (nullable placeholders)
    df.insert(0, "Unnamed: 0", range(len(df)))
    df.insert(1, "description", "Sample transcription")
    df.insert(3, "sample_name", "Sample " + df.index.astype(str))
    df.insert(4, "keywords", "")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} synthetic samples to {out}")
    return df


if __name__ == "__main__":
    generate_sample_dataset()
