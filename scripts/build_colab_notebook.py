"""Generate notebooks/medguard_model_server_colab.ipynb with serve/model_server.py embedded verbatim,
so the notebook needs no repo checkout and never drifts from the server code.

    python scripts/build_colab_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server_src = open(os.path.join(ROOT, "serve", "model_server.py")).read()

CELLS = [
    ("md", """# MedGuard model server (Colab T4)
Runs **only MedGemma-4B-it (4-bit)** behind a token-protected API, exposed through ngrok. The guardrails and UI run on your laptop.

**One-time setup**
1. Runtime → Change runtime type → **T4 GPU**.
2. Accept the MedGemma terms at huggingface.co/google/medgemma-4b-it.
3. Colab **Secrets** (key icon): `HF_TOKEN`, `NGROK_TOKEN` (dashboard.ngrok.com), `MEDGUARD_TOKEN` (any long random string), and optionally `NGROK_DOMAIN` (your free static domain, so the URL survives restarts). Turn on notebook access for each.

Then **Runtime → Run all**. The last cell prints the line to paste on your laptop."""),
    ("code", "!pip install -q unsloth fastapi uvicorn pyngrok"),
    ("code", "%%writefile model_server.py\n" + server_src),
    ("code", """import os
from google.colab import userdata
from huggingface_hub import login

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

login(secret("HF_TOKEN"))
TOKEN = secret("MEDGUARD_TOKEN")
assert TOKEN and len(TOKEN) >= 16, "add a MEDGUARD_TOKEN secret (16+ random characters)"
assert secret("NGROK_TOKEN"), "add an NGROK_TOKEN secret"
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"""),
    ("code", """import model_server
model = model_server.MedGemma(adapter=None)   # base MedGemma; set adapter="medgemma-medmcqa-lora" once fine-tuned
app = model_server.create_app(model, TOKEN)
server = model_server.start_in_background(app, 8000)
print("server started:", server.started)"""),
    ("code", """from pyngrok import ngrok
ngrok.set_auth_token(secret("NGROK_TOKEN"))
domain = secret("NGROK_DOMAIN")
URL = (ngrok.connect(8000, "http", domain=domain) if domain else ngrok.connect(8000, "http")).public_url
print(URL)"""),
    ("code", """# Smoke test through the tunnel: auth, text generation, image generation
import requests, time, io, base64
from PIL import Image
H = {"Authorization": f"Bearer {TOKEN}", "ngrok-skip-browser-warning": "1"}
assert requests.get(f"{URL}/health", headers={"ngrok-skip-browser-warning": "1"}).status_code == 401
print(requests.get(f"{URL}/health", headers=H).json())
t = time.time()
r = requests.post(f"{URL}/generate", headers=H, json={"system": "Be brief.", "user": "Name three common causes of fever.", "max_new_tokens": 60}).json()
print(f"text  {time.time()-t:.1f}s:", r["text"][:200])
buf = io.BytesIO(); Image.new("RGB", (256, 256), (40, 40, 40)).save(buf, "JPEG")
t = time.time()
r = requests.post(f"{URL}/generate", headers=H, json={"system": "Describe images briefly.", "user": "What do you see?", "image_b64": base64.b64encode(buf.getvalue()).decode(), "max_new_tokens": 40}).json()
print(f"image {time.time()-t:.1f}s:", r["text"][:200])"""),
    ("code", """print("Paste on your laptop (macOS/Linux):\\n")
print(f"export MEDGUARD_REMOTE_URL={URL}")
print(f"export MEDGUARD_REMOTE_TOKEN={TOKEN}")
print("python app.py")"""),
    ("md", """Keep this tab open. Free Colab disconnects after roughly 90 minutes idle. If it does, **Run all** again. With `NGROK_DOMAIN` and `MEDGUARD_TOKEN` stored as secrets, the laptop config stays the same."""),
]


def cell(kind, src):
    lines = [l + "\n" for l in src.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {"cells": [cell(k, s) for k, s in CELLS],
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "notebooks", "medguard_model_server_colab.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
