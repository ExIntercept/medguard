"""MedGemma model server (GPU side). Self-contained: needs only this file, no repo checkout.

    MEDGUARD_TOKEN=<secret> python model_server.py

Env:
    MEDGUARD_TOKEN    bearer token the laptop must send (generated if unset, printed at start)
    MEDGUARD_MODEL    base model (default: unsloth/medgemma-4b-it-unsloth-bnb-4bit, 4-bit, fits a T4)
    MEDGUARD_ADAPTER  optional LoRA adapter dir from extras/finetune_medmcqa.ipynb

Endpoints (all require `Authorization: Bearer <token>`):
    GET  /health    -> status, model, adapter loaded, GPU, VRAM
    POST /generate  -> {"text", "latency_s"}   body: system, user, image_b64?, use_adapter, max_new_tokens

Unsloth is used to load Gemma 3 on a T4: the T4 has no native bf16 and plain fp16 Gemma 3
overflows; Unsloth patches that. Generation is serialised with a lock (one model, one GPU).
"""
from __future__ import annotations

import base64
import io
import os
import secrets
import threading
import time
from contextlib import nullcontext

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

DEFAULT_MODEL = "unsloth/medgemma-4b-it-unsloth-bnb-4bit"


class MedGemma:
    """One 4-bit MedGemma. With an adapter loaded, use_adapter=False gives the original model
    (VLM role) and use_adapter=True the fine-tuned one (LLM role); without one, both are the base."""

    def __init__(self, model_id: str = DEFAULT_MODEL, adapter: str | None = None, max_seq_length: int = 4096):
        from unsloth import FastModel
        self.model_id, self.adapter = model_id, adapter
        self.model, self.proc = FastModel.from_pretrained(
            model_name=adapter or model_id, max_seq_length=max_seq_length, load_in_4bit=True)
        FastModel.for_inference(self.model)
        self.has_adapter = bool(adapter) and hasattr(self.model, "disable_adapter")

    def generate(self, system: str, user: str, image=None, use_adapter: bool = True, max_new_tokens: int = 512) -> str:
        import torch
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": user}]
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        prompt = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        kw = {"text": prompt, "add_special_tokens": False, "return_tensors": "pt"}
        if image is not None:
            kw["images"] = image
        inputs = self.proc(**kw).to(self.model.device)
        ctx = self.model.disable_adapter() if (self.has_adapter and not use_adapter) else nullcontext()
        with ctx, torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class GenReq(BaseModel):
    system: str = Field(max_length=10_000)
    user: str = Field(max_length=20_000)
    image_b64: str | None = Field(default=None, max_length=8_000_000)
    use_adapter: bool = True
    max_new_tokens: int = Field(default=512, ge=1, le=1024)


def create_app(model, token: str) -> FastAPI:
    app = FastAPI(title="MedGuard model server")
    app.state.model, app.state.lock = model, threading.Lock()

    def auth(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health", dependencies=[Depends(auth)])
    def health(request: Request):
        m = request.app.state.model
        info = {"status": "ok", "model": getattr(m, "model_id", "?"), "adapter": getattr(m, "has_adapter", False)}
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["vram_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        except ImportError:
            pass
        return info

    @app.post("/generate", dependencies=[Depends(auth)])
    def generate(req: GenReq, request: Request):
        image = None
        if req.image_b64:
            from PIL import Image
            try:
                image = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
            except Exception:
                raise HTTPException(status_code=400, detail="could not decode image")
        t0 = time.perf_counter()
        with request.app.state.lock:
            text = request.app.state.model.generate(req.system, req.user, image=image,
                                                    use_adapter=req.use_adapter, max_new_tokens=req.max_new_tokens)
        return {"text": text, "latency_s": round(time.perf_counter() - t0, 2)}

    return app


def start_in_background(app: FastAPI, port: int = 8000):
    """For notebooks: run uvicorn in a daemon thread so the cell returns."""
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    return server


if __name__ == "__main__":
    import uvicorn
    token = os.getenv("MEDGUARD_TOKEN") or secrets.token_urlsafe(24)
    model = MedGemma(os.getenv("MEDGUARD_MODEL", DEFAULT_MODEL), os.getenv("MEDGUARD_ADAPTER") or None)
    print("MEDGUARD_TOKEN =", token)
    uvicorn.run(create_app(model, token), host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
