from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: dict):
    return {
        "input_text": data.get("note", ""),
        "CPT_suggestions": ["99213", "99214", "99215"],
        "confidence": [0.8, 0.15, 0.05]
    }