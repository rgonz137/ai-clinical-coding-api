from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import re

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Feature extraction (copied from your notebook)
def extract_clinical_features(note):
    text = note.lower()

    def check(pos, neg):
        for n in neg:
            if re.search(n, text):
                return 0
        for p in pos:
            if re.search(p, text):
                return 1
        return 0

    features = {
        "CXR_Pos": check([r'infiltrate', r'consolidation', r'cxr', r'x-ray'], [r'no infiltrate', r'clear']),
        "Fever": check([r'fever', r'100', r'102', r'103'], [r'afebrile', r'no fever']),
        "Hypoxia": check([r'hypox', r'o2\s*sat\s*<\s*90'], [r'room air', r'98%', r'99%']),
        "Confusion": check([r'confusion', r'ams'], [r'alert', r'oriented']),
        "Age": 50,  # default if not provided
        "Is_ED": 0,
        "Is_Inpatient": 1
    }

    return pd.DataFrame([features])


class ClinicalNote(BaseModel):
    note: str


@app.post("/predict")
def predict(note: ClinicalNote):
    features_df = extract_clinical_features(note.note)

    pred = model.predict(features_df)[0]

    return {
        "input": note.note,
        "prediction": pred
    }