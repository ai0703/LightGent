"""Serve a 4B model (tuned or stock) on Modal for the eval harness.

Bounded-run only: deploy, evaluate, then
    python -m modal app stop lightgent-eval

Which model gets served is read at DEPLOY time from finetune/data/eval_model.txt,
so the same app can serve the tuned model and then the stock baseline:

    echo /vol/merged/qwen3-4b-lightgent-v1 > finetune/data/eval_model.txt
    python -m modal deploy finetune/serve_eval_modal.py
    ... run eval ...
    echo Qwen/Qwen3-4B-Instruct-2507 > finetune/data/eval_model.txt
    python -m modal deploy finetune/serve_eval_modal.py

URL: https://automatesystem1--lightgent-eval-serve.modal.run/v1
Cost: L40S about 1.95 USD/hr, billed per second, scale-to-zero after 5 min idle.
A 4B model needs roughly 8 GB of the card's 48 GB, so there is ample KV cache.
"""
import os
import subprocess
from pathlib import Path

import modal

GPU = "L40S"
PORT = 8000
MAX_MODEL_LEN = 16384
TOOL_PARSER = "hermes"

_data_dir = Path(__file__).parent / "data"


def _read_file(name: str) -> str | None:
    path = _data_dir / name
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


_MODEL = os.environ.get("EVAL_MODEL") or _read_file("eval_model.txt")
_KEY = os.environ.get("TEACHER_API_KEY") or _read_file("teacher.key")

if modal.is_local():
    if not _MODEL:
        raise RuntimeError("finetune/data/eval_model.txt is missing or empty")
    if not _KEY:
        raise RuntimeError("finetune/data/teacher.key is missing or empty")

# Same pins as the verified smoke image in train_modal.py: vLLM 0.8.5 calls a
# tokenizer API that transformers 5.x removed, so transformers must stay on 4.x.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.8.5.post1",
        "transformers==4.53.2",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
finetune_vol = modal.Volume.from_name("lightgent-finetune", create_if_missing=True)

app = modal.App("lightgent-eval")


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=300,
    timeout=60 * 60,
    secrets=[
        modal.Secret.from_dict(
            {"TEACHER_API_KEY": _KEY or "", "EVAL_MODEL": _MODEL or ""}
        )
    ],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/vol": finetune_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    model = (os.environ.get("EVAL_MODEL") or "").strip()
    key = (os.environ.get("TEACHER_API_KEY") or "").strip()
    if not model or not key:
        raise RuntimeError("EVAL_MODEL or TEACHER_API_KEY empty in container")
    cmd = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--api-key", key,
        "--served-model-name", "eval-model",
        "--enable-auto-tool-choice",
        "--tool-call-parser", TOOL_PARSER,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
    ]
    subprocess.Popen(cmd)
