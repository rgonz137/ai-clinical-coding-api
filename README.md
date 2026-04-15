# AI Clinical Coding Assistant — Backend

FastAPI backend that predicts CPT codes from a free-text clinical note using a
TF-IDF vectorizer + classifier trained in Colab.

## Files in the repo

| File | Purpose |
|------|---------|
| `api.py` | FastAPI app. Downloads artifacts from Hugging Face on startup. |
| `requirements.txt` | Pinned to match the Colab training environment. |
| `runtime.txt` | Python 3.12.7 (required for scipy 1.16.3). |
| `Procfile` | Render start command. |
| `.gitignore` | Excludes the model artifacts (too big for GitHub). |

## NOT in the repo (by design)

- `model.pkl`, `vectorizer.pkl`, `cpt_encoder.pkl` — hosted on Hugging Face:
  https://huggingface.co/rgonz137/ai-clinical-coding-model
- `app.py` — Streamlit demo, not used by this service.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 10000
```

On first run the service downloads the three artifacts from Hugging Face into
`./artifacts/`. Subsequent runs use the cache.

Visit http://localhost:10000/docs for Swagger UI.

## Endpoints

`GET /health`
```json
{"status": "ok", "model_loaded": true, "hf_repo": "rgonz137/ai-clinical-coding-model"}
```

`POST /predict`
```json
// request
{"note": "72 y/o with fever, cough, RLL infiltrate on CXR", "top_n": 3}

// response
{
  "input_text": "...",
  "predictions": [
    {"cpt_code": "99223", "confidence": 0.71},
    {"cpt_code": "71046", "confidence": 0.18},
    {"cpt_code": "99214", "confidence": 0.06}
  ]
}
```

## Deploy to Render

1. Push this repo to GitHub (model files are gitignored; artifacts come from HF).
2. Render → New → Web Service → connect repo.
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn api:app --host 0.0.0.0 --port $PORT` (Procfile handles this).
5. Optional env var: `HF_REPO=rgonz137/ai-clinical-coding-model` (already the default).
6. First cold start takes ~30-60s (downloading 125 MB from HF). Hit `/health`
   until `model_loaded: true` before calling `/predict`.

## Frontend integration (Jiahao)

- Base URL: your Render service URL
- CORS is open (`*`) — tighten to the deployed frontend origin before the demo
- Schema above is stable — do not rename `predictions`, `cpt_code`, `confidence`
- Ping `/health` on page load to warm the cold-start

## Why this setup

- **Artifacts on Hugging Face** — `model.pkl` is 125 MB, exceeds GitHub's 100 MB
  limit. HF's free tier handles ML artifacts natively with a proper CDN.
- **Python 3.12** — matches Colab training (3.12.13). scipy 1.16.3 requires
  Python ≥ 3.11, so 3.10 is not viable.
- **numpy 2.0.2 pinned** — pickles made with numpy 2.x can fail to load under
  numpy 1.x. This was a likely cause of the prior `KeyError` errors.
- **`lifespan` startup** — artifacts load once at boot, not per-request.
