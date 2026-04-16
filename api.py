"""
AI Clinical Coding Assistant — FastAPI backend

Response schema matches JUDY.AI Integration Guidelines:
    {
      "cptSuggestions":   [{"code": "71045", "label": "Chest X-ray", "confidence": 0.88}],
      "icd10Suggestions": [{"code": "J18.9", "label": "Pneumonia, unspecified", "confidence": 0.62}],
      "revenueCycleNotes": ["Documentation supports ..."]
    }

Pipeline:
  clinical note (free text)
    -> TfidfVectorizer.transform
    -> RandomForestClassifier.predict_proba  (60 CPT classes)
    -> LabelEncoder.inverse_transform         (integer index -> CPT code)
    -> cpt_lookup.json                        (CPT label + associated ICD-10 codes)

Model artifacts download from Hugging Face at startup.
CPT/ICD-10 lookup table ships in the repo at ./data/cpt_lookup.json.
"""

import json
import os
import sys
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
HF_REPO = os.environ.get("HF_REPO", "rgonz137/ai-clinical-coding-model")
ARTIFACTS = ["model.pkl", "vectorizer.pkl", "cpt_encoder.pkl"]
LOCAL_DIR = Path(os.environ.get("ARTIFACT_DIR", str(BASE_DIR / "artifacts")))
LOOKUP_PATH = BASE_DIR / "data" / "cpt_lookup.json"


# ---------- helpers ----------

def format_icd(code: str) -> str:
    """Insert a dot after the 3rd character if not already present (e.g. J189 -> J18.9)."""
    code = (code or "").strip()
    if not code or "." in code or len(code) <= 3:
        return code
    return code[:3] + "." + code[3:]


def fetch_artifacts() -> dict[str, Path]:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ARTIFACTS:
        local = LOCAL_DIR / name
        if local.exists() and local.stat().st_size > 0:
            print(f"[startup] {name} cached at {local}", flush=True)
        else:
            print(f"[startup] downloading {name} from {HF_REPO}...", flush=True)
            downloaded = hf_hub_download(repo_id=HF_REPO, filename=name, local_dir=str(LOCAL_DIR))
            local = Path(downloaded)
            print(f"[startup] downloaded {name} ({local.stat().st_size / 1e6:.1f} MB)", flush=True)
        paths[name] = local
    return paths


def load_lookup() -> dict:
    if not LOOKUP_PATH.exists():
        print(f"[startup] WARNING: {LOOKUP_PATH} not found — labels will be empty",
              file=sys.stderr, flush=True)
        return {}
    with open(LOOKUP_PATH) as f:
        return json.load(f)


# ---------- lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        paths = fetch_artifacts()
        app.state.model = joblib.load(paths["model.pkl"])
        app.state.vectorizer = joblib.load(paths["vectorizer.pkl"])
        app.state.encoder = joblib.load(paths["cpt_encoder.pkl"])
        app.state.lookup = load_lookup()
        app.state.ready = True
        print(
            f"[startup] loaded model (n_features_in_={app.state.model.n_features_in_}, "
            f"n_classes_={app.state.model.n_classes_}) + lookup ({len(app.state.lookup)} CPT entries)",
            flush=True,
        )
    except Exception as e:
        app.state.ready = False
        app.state.load_error = f"{type(e).__name__}: {e}"
        print(f"[startup] FAILED: {app.state.load_error}", file=sys.stderr, flush=True)
    yield


app = FastAPI(
    title="AI Clinical Coding Assistant",
    description="Predicts CPT codes and related ICD-10 codes from a clinical note.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------- schemas ----------

class ClinicalNote(BaseModel):
    note: str = Field(..., min_length=1, description="Free-text clinical note")
    top_n: int = Field(3, ge=1, le=10, description="How many CPT suggestions to return")


class Suggestion(BaseModel):
    code: str
    label: str
    confidence: float


class PredictResponse(BaseModel):
    input_text: str
    cptSuggestions: list[Suggestion]
    icd10Suggestions: list[Suggestion]
    revenueCycleNotes: list[str]


# ---------- core predict logic ----------

def build_response(note: str, top_n: int) -> PredictResponse:
    vec = app.state.vectorizer.transform([note])
    probs = app.state.model.predict_proba(vec)[0]
    top_idx = np.argsort(probs)[::-1][:top_n]
    cpt_codes_ordered = app.state.encoder.inverse_transform(top_idx)
    lookup = app.state.lookup

    # --- CPT suggestions ---
    cpt_suggestions: list[Suggestion] = []
    for i, cpt_raw in zip(top_idx, cpt_codes_ordered):
        cpt_str = str(cpt_raw)
        entry = lookup.get(cpt_str, {})
        cpt_suggestions.append(Suggestion(
            code=cpt_str,
            label=entry.get("label", f"CPT {cpt_str}"),
            confidence=round(float(probs[i]), 4),
        ))

    # --- ICD-10 suggestions (derived from top CPTs weighted by CPT confidence x ICD support) ---
    icd_scores: dict[str, float] = defaultdict(float)
    icd_labels: dict[str, str] = {}
    for sug in cpt_suggestions:
        entry = lookup.get(sug.code, {})
        for icd in entry.get("related_icd10", []):
            code = icd["code"]
            score = sug.confidence * float(icd.get("support", 0.0))
            icd_scores[code] = max(icd_scores[code], score)
            icd_labels[code] = icd.get("label", code)

    icd_ranked = sorted(icd_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    icd10_suggestions = [
        Suggestion(code=format_icd(c), label=icd_labels.get(c, c), confidence=round(score, 4))
        for c, score in icd_ranked
    ]

    # --- Revenue cycle notes ---
    if cpt_suggestions and cpt_suggestions[0].confidence >= 0.25:
        top = cpt_suggestions[0]
        revenue_notes = [
            f"Documentation supports CPT {top.code} ({top.label}) with confidence {top.confidence:.0%}.",
        ]
        if icd10_suggestions:
            top_icd = icd10_suggestions[0]
            revenue_notes.append(
                f"Primary diagnosis aligns with ICD-10 {top_icd.code} ({top_icd.label})."
            )
    else:
        revenue_notes = [
            "Low model confidence on primary code — recommend manual coder review before submission.",
        ]

    return PredictResponse(
        input_text=note,
        cptSuggestions=cpt_suggestions,
        icd10Suggestions=icd10_suggestions,
        revenueCycleNotes=revenue_notes,
    )


# ---------- endpoints ----------

@app.get("/")
def root():
    return {
        "service": "AI Clinical Coding Assistant",
        "version": "1.1.0",
        "ready": getattr(app.state, "ready", False),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    if getattr(app.state, "ready", False):
        return {
            "status": "ok",
            "model_loaded": True,
            "hf_repo": HF_REPO,
            "cpt_lookup_entries": len(app.state.lookup),
        }
    return {
        "status": "not_ready",
        "model_loaded": False,
        "error": getattr(app.state, "load_error", "still loading"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: ClinicalNote):
    if not getattr(app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {getattr(app.state, 'load_error', 'still loading')}",
        )
    try:
        return build_response(payload.note, payload.top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), reload=False)
