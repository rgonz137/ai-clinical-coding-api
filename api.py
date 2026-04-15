"""
AI Clinical Coding Assistant — FastAPI backend

Pipeline: clinical note (free text)
    -> TfidfVectorizer.transform
    -> model.predict_proba
    -> LabelEncoder.inverse_transform  (returns CPT codes)

Model artifacts are hosted on Hugging Face and downloaded at startup.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

HF_REPO = os.environ.get("HF_REPO", "rgonz137/ai-clinical-coding-model")
ARTIFACTS = ["model.pkl", "vectorizer.pkl", "cpt_encoder.pkl"]
LOCAL_DIR = Path(os.environ.get("ARTIFACT_DIR", "./artifacts"))


# ---------- artifact download ----------

def fetch_artifacts() -> dict[str, Path]:
    """Download artifacts from HF Hub (cached) and return local paths."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ARTIFACTS:
        local = LOCAL_DIR / name
        if local.exists() and local.stat().st_size > 0:
            print(f"[startup] {name} cached at {local}", flush=True)
        else:
            print(f"[startup] downloading {name} from {HF_REPO}...", flush=True)
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename=name,
                local_dir=str(LOCAL_DIR),
                local_dir_use_symlinks=False,
            )
            local = Path(downloaded)
            print(f"[startup] downloaded {name} -> {local} "
                  f"({local.stat().st_size / 1e6:.1f} MB)", flush=True)
        paths[name] = local
    return paths


# ---------- app lifespan: load everything once at startup ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        paths = fetch_artifacts()
        app.state.model = joblib.load(paths["model.pkl"])
        app.state.vectorizer = joblib.load(paths["vectorizer.pkl"])
        app.state.encoder = joblib.load(paths["cpt_encoder.pkl"])
        app.state.ready = True
        print(f"[startup] model loaded. "
              f"n_features_in_={getattr(app.state.model, 'n_features_in_', '?')}, "
              f"n_classes_={getattr(app.state.model, 'n_classes_', '?')}",
              flush=True)
    except Exception as e:
        app.state.ready = False
        app.state.load_error = f"{type(e).__name__}: {e}"
        print(f"[startup] FAILED to load artifacts: {app.state.load_error}",
              file=sys.stderr, flush=True)
        # Don't raise — /health can report the failure instead of crashing the process
    yield


app = FastAPI(
    title="AI Clinical Coding Assistant",
    description="Predicts CPT codes from a clinical note.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to Jiahao's frontend URL before final demo
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------- schemas ----------

class ClinicalNote(BaseModel):
    note: str = Field(..., min_length=1, description="Free-text clinical note")
    top_n: int = Field(3, ge=1, le=10, description="How many CPT suggestions to return")


class CPTSuggestion(BaseModel):
    cpt_code: str
    confidence: float


class PredictResponse(BaseModel):
    input_text: str
    predictions: list[CPTSuggestion]


# ---------- endpoints ----------

@app.get("/")
def root():
    return {
        "service": "AI Clinical Coding Assistant",
        "status": "ok",
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
        vec = app.state.vectorizer.transform([payload.note])
        probs = app.state.model.predict_proba(vec)[0]
        top_idx = np.argsort(probs)[::-1][: payload.top_n]
        cpt_codes = app.state.encoder.inverse_transform(top_idx)

        predictions = [
            CPTSuggestion(cpt_code=str(code), confidence=round(float(probs[i]), 4))
            for code, i in zip(cpt_codes, top_idx)
        ]
        return PredictResponse(input_text=payload.note, predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
