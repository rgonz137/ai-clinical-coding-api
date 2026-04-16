# AI Clinical Coding Assistant — Backend

FastAPI backend that predicts CPT codes and related ICD-10 codes from a free-text clinical note.
Response schema matches the JUDY.AI Integration Guidelines (`cptSuggestions` / `icd10Suggestions` / `revenueCycleNotes`).

## Files in the repo

| File | Purpose |
|------|---------|
| `api.py` | FastAPI app. Downloads model from Hugging Face on startup, loads local `data/cpt_lookup.json` for labels. |
| `data/cpt_lookup.json` | CPT code → English label + associated ICD-10 codes. Built from the team's training dataset. |
| `requirements.txt` | Pinned to match Colab training environment. |
| `runtime.txt` | Python 3.12.7 (required for scipy 1.16.3). |
| `Procfile` | Render start command. |

## NOT in the repo (by design)

- `model.pkl`, `vectorizer.pkl`, `cpt_encoder.pkl` — hosted at https://huggingface.co/rgonz137/ai-clinical-coding-model
- `app.py` — Streamlit demo, replaced by this API.

## Response shape (v1.1 — matches JUDY.AI integration spec)

```json
POST /predict
Request:  {"note": "72 y/o with fever and RLL infiltrate on CXR", "top_n": 3}

Response:
{
  "input_text": "72 y/o with fever and RLL infiltrate on CXR",
  "cptSuggestions": [
    {"code": "71010", "label": "Chest X-ray, single view", "confidence": 0.45}
  ],
  "icd10Suggestions": [
    {"code": "J18.9", "label": "Pneumonia, unspecified organism", "confidence": 0.45}
  ],
  "revenueCycleNotes": [
    "Documentation supports CPT 71010 (Chest X-ray, single view) with confidence 45%.",
    "Primary diagnosis aligns with ICD-10 J18.9 (Pneumonia, unspecified organism)."
  ]
}
```

Notes on ICD-10: the underlying model only predicts CPT codes. ICD-10 suggestions are derived from the top CPT predictions, weighted by (CPT confidence) × (historical ICD-10/CPT co-occurrence in training data).

## Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 10000
```

First run downloads the three artifacts from Hugging Face into `./artifacts/`. Subsequent runs use the cache.

- Swagger UI: http://localhost:10000/docs
- Health: http://localhost:10000/health

## Deploy to Render

Push to `main` → Render auto-redeploys. Start command is in `Procfile`.

Environment — make sure **no `PYTHON_VERSION` env var** is set in the Render dashboard (it overrides `runtime.txt`). If set, delete it so 3.12.7 from `runtime.txt` takes effect.

## Re-training flow

1. Retrain in Colab, run `joblib.dump(...)` cell.
2. Upload the three new `.pkl` files to https://huggingface.co/rgonz137/ai-clinical-coding-model (delete old, upload new).
3. In Render dashboard → **Manual Deploy → Clear build cache & deploy**. Service re-downloads the new model from Hugging Face on startup.

No code push needed.

## Compliance with JUDY.AI integration spec

This backend satisfies **Group 1 – Model Team** deliverables:
- Input format defined (`{note, top_n}`)
- Output JSON defined and live (`cptSuggestions` / `icd10Suggestions` / `revenueCycleNotes`)
- Sample output available at `/docs`

Group 2 (Integration Team) can consume this endpoint directly from `src/app/api/coding/suggestions/route.ts` — the response shape requires no adapter in `mapCodingResponse.ts`.
