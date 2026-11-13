import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Hugging Face থেকে আপনার স্থায়ী মডেল লোড করা
model_repo = "Joysarker518/bangla-sms-model"
hf_token = os.environ.get("HF_TOKEN")

classifier = pipeline(
    "text-classification",
    model=model_repo,
    tokenizer="xlm-roberta-base",
    token=hf_token,
    truncation=True,
    max_length=128
)

class SMSRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "Bangla SMS Classifier API is running permanently!"}

@app.post("/predict")
def predict_sms(data: SMSRequest):
    if not data.message.strip():
        return {"label": "ERROR", "confidence": 0.0}
    res = classifier(data.message)[0]
    return {
        "label": res['label'].upper(),
        "confidence": round(res['score'] * 100, 2)
    }
