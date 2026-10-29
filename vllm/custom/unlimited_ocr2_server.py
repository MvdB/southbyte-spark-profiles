#!/usr/bin/env python3
"""OpenAI-kompatibler Chat-Endpunkt für baidu/Unlimited-OCR-2 über transformers.

Unlimited-OCR-2 läuft (Stand 29.10.2026) in keiner vLLM-Version: die DASH-Attention
steckt in einem offenen PR (vllm#64016), recipes.vllm.ai führt nur Version 1. Die
Modellkarte nennt zwei Wege, transformers (getestet mit 4.52.1) und ein eigenes
SGLang-Rad. Hier der transformers-Weg, eine Anfrage nach der anderen:

  POST /v1/chat/completions   Bild als image_url (data-URI), der Text wird nicht
                              ausgewertet — die Aufgabe ist fest ("Multi page parsing.")
  GET  /v1/models, /health

Dekodierung wie die Karte bzw. das V1-Rezept für eine Seite: gierig, N-Gramm-Sperre
35 über ein Fenster von 128, Seitenwache an; crop_mode automatisch (Kachelung bei einer
Seite). Die Antwort ist die Rohausgabe mit Sondertoken (<PAGE>, <BLOCK>, <FORK> …).
"""
from __future__ import annotations

import base64
import io
import os
import time

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODELL = os.environ.get("UOCR2_PATH", "/hf_models/baidu--Unlimited-OCR-2")
NAME = os.environ.get("SERVED_NAME", "Unlimited-OCR-2")
MAX_NEU = int(os.environ.get("UOCR2_MAX_NEW_TOKENS", "8192"))

app = FastAPI(title="unlimited-ocr2", version="0.1.0")
tok = AutoTokenizer.from_pretrained(MODELL, trust_remote_code=True)
modell = AutoModel.from_pretrained(MODELL, trust_remote_code=True, use_safetensors=True,
                                   torch_dtype=torch.bfloat16).eval().cuda()


def _bild(nachricht: dict) -> Image.Image:
    teile = nachricht.get("content")
    for t in teile if isinstance(teile, list) else []:
        if t.get("type") == "image_url":
            url = t["image_url"]["url"]
            if not url.startswith("data:"):
                raise HTTPException(400, "nur data-URI")
            return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    raise HTTPException(400, "kein Bild in der Anfrage")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODELL}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": NAME, "object": "model", "max_model_len": 131072}]}


@app.post("/v1/chat/completions")
def chat(anfrage: dict) -> dict:
    bild = _bild(anfrage["messages"][-1])
    grenze = min(MAX_NEU, int(anfrage.get("max_tokens") or MAX_NEU))
    t0 = time.time()
    with torch.no_grad():
        erg = modell.infer(tok, image_file=bild, eval_mode=True, temperature=0.0,
                           no_repeat_ngram_size=35, ngram_window=128, max_new_tokens=grenze,
                           return_dict=True)
    text = erg["text"] if isinstance(erg, dict) else erg
    n = len(tok.encode(text, add_special_tokens=False))
    grund = "length" if n >= grenze else "stop"
    return {"id": f"uocr2-{int(t0 * 1000)}", "object": "chat.completion", "created": int(t0), "model": NAME,
            "choices": [{"index": 0, "finish_reason": grund,
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": n, "total_tokens": n},
            "gen_seconds": round(time.time() - t0, 2)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
