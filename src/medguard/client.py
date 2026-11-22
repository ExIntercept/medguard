"""Laptop-side client for the MedGemma model server (serve/model_server.py, running on Colab).

The guardrails never leave the laptop: the GPU box only receives prompts that have already
passed the input guard and the taint scan, and its outputs come back here for the output guard.
"""
from __future__ import annotations

import base64
import io
import os

import requests


class ModelServerError(RuntimeError):
    pass


class ModelClient:
    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = 180.0):
        url = url or os.getenv("MEDGUARD_REMOTE_URL")
        token = token or os.getenv("MEDGUARD_REMOTE_TOKEN")
        if not url or not token:
            raise ModelServerError("set MEDGUARD_REMOTE_URL and MEDGUARD_REMOTE_TOKEN (printed by the Colab notebook)")
        self.url, self.timeout = url.rstrip("/"), timeout
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "ngrok-skip-browser-warning": "1"})  # bypass ngrok free-tier interstitial

    def _call(self, method: str, path: str, **kw) -> dict:
        try:
            r = self.s.request(method, f"{self.url}{path}", **kw)
        except requests.RequestException as e:
            raise ModelServerError(f"model server unreachable ({e.__class__.__name__}). Is the Colab cell still running?") from e
        if r.status_code == 401:
            raise ModelServerError("model server rejected the token (401)")
        if r.status_code >= 400:
            raise ModelServerError(f"model server error {r.status_code}: {r.text[:200]}")
        return r.json()

    def health(self) -> dict:
        return self._call("GET", "/health", timeout=15)

    def generate(self, system: str, user: str, image=None, use_adapter: bool = True,
                 max_new_tokens: int = 512) -> str:
        payload = {"system": system, "user": user, "use_adapter": use_adapter, "max_new_tokens": max_new_tokens}
        if image is not None:
            im = image.convert("RGB")
            im.thumbnail((896, 896))  # MedGemma's vision encoder works at 896 px
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            payload["image_b64"] = base64.b64encode(buf.getvalue()).decode()
        return self._call("POST", "/generate", json=payload, timeout=self.timeout)["text"]
