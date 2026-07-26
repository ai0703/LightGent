"""Serve openai/gpt-oss-120b on Modal as the TEACHER endpoint for trajectory banking.

This is a BOUNDED-RUN exception to the standing no-Modal-serving rule,
approved by Abdul on 2026-07-26 after the free Mistral pool died (all keys
401). Teacher upgraded from Qwen3-30B-A3B to gpt-oss-120b by Abdul's choice.

Deploy:      python -m modal deploy finetune/serve_teacher_modal.py
URL:         https://automatesystem1--lightgent-teacher-serve.modal.run/v1
Shutdown:    python -m modal app stop lightgent-teacher

Point the banking driver at it:
    python -m finetune.bank_trajectories \
        --base-url https://automatesystem1--lightgent-teacher-serve.modal.run/v1 \
        --model openai/gpt-oss-120b \
        --api-key <contents of finetune/data/teacher.key>

Cost math: H100 is roughly 3.95 USD per hour, billed per second. A bounded
banking run is 2 to 4 hours, so roughly 8 to 16 USD, expected within the
Modal starter credits. The endpoint scales to zero after 5 minutes idle and
MUST be stopped with the shutdown command when banking finishes.

API key: read at deploy time from finetune/data/teacher.key (gitignored,
never hardcoded, never printed). Inside the container the key arrives as the
TEACHER_API_KEY env var via an inline Modal secret.
"""
import os
import subprocess
from pathlib import Path

import modal

MODEL = "openai/gpt-oss-120b"
GPU = "H100"
PORT = 8000
MAX_MODEL_LEN = 16384

# vllm 0.10.1.1 has first-class gpt-oss support (harmony format) and the
# "openai" tool-call parser that returns standard OpenAI tool_calls.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.10.1.1",
        "transformers==4.55.2",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("lightgent-teacher")


def _read_teacher_key() -> str | None:
    env_val = (os.environ.get("TEACHER_API_KEY") or "").strip()
    if env_val:
        return env_val
    key_file = Path(__file__).parent / "data" / "teacher.key"
    if key_file.exists():
        file_val = key_file.read_text(encoding="utf-8").strip()
        if file_val:
            return file_val
    return None


_TEACHER_KEY = _read_teacher_key()

if modal.is_local() and not _TEACHER_KEY:
    raise RuntimeError(
        "finetune/data/teacher.key is missing or empty. Create it (one line, "
        "the API key for this endpoint) before deploying."
    )


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=300,
    timeout=60 * 60,
    secrets=[modal.Secret.from_dict({"TEACHER_API_KEY": _TEACHER_KEY or ""})],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=25 * 60)
def serve():
    key = (os.environ.get("TEACHER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("TEACHER_API_KEY is empty inside the container")
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--api-key", key,
        "--served-model-name", MODEL,
        "--enable-auto-tool-choice",
        "--tool-call-parser", "openai",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.92",
    ]
    subprocess.Popen(cmd)
