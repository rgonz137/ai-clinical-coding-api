from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
cpt_encoder = joblib.load("cpt_encoder.pkl")

app = FastAPI()

class ClinicalNote(BaseModel):
    note: str

def get_top_cpt_predictions(note, top_n=3):
    vec = vectorizer.transform([note])
    probs = model.predict_proba(vec)[0]

    top_indices = np.argsort(probs)[::-1][:top_n]

    return {
        "input_text": note,
        "CPT_suggestions": cpt_encoder.inverse_transform(top_indices).tolist(),
        "confidence": [float(round(probs[i], 3)) for i in top_indices]
    }

@app.post("/predict")
def predict(note: ClinicalNote):
    return get_top_cpt_predictions(note.note)