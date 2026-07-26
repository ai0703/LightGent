"""OpenAI-compatible shim that serves Codex (gpt-5.6-sol) and Claude Haiku
as teacher models through their CLIs.

Why this exists: the banking pipeline (lightgent_service + bank_trajectories)
speaks the OpenAI chat-completions protocol, but the two teachers we can
afford bill to flat-rate subscriptions and only exist as CLIs:

    codex exec  -> gpt-5.6-sol   (ChatGPT Plus, $0 marginal)
    claude -p   -> Haiku          (Claude subscription, cheap)

The shim translates /chat/completions into a single CLI invocation. The
teacher never emits hermes tool-call syntax; it replies with a tiny JSON
decision object and the shim converts that into OpenAI tool_calls, so the
trajectories logged downstream are in exactly the training schema.

Run:
    python -m finetune.cli_teacher_shim --port 8944
Then:
    LLM_BASE_URL=http://127.0.0.1:8944/v1  LLM_MODEL=sol   (or haiku)

Model routing: request model containing "sol" or "codex" -> codex exec;
containing "haiku" or "claude" -> claude -p. Anything else 400s loudly, no
silent downgrades (standing rule).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

log = logging.getLogger("cli-teacher-shim")

app = FastAPI()

# One semaphore per backend: codex cold-starts a session per call, so keep it
# modest; claude -p is lighter.
LIMITS = {"codex": asyncio.Semaphore(3), "claude": asyncio.Semaphore(4)}
CALL_TIMEOUT = 240  # seconds per teacher decision

DECISION_INSTRUCTIONS = """\
You are the decision engine inside a web-research agent. Read the transcript
above. Reply with ONE JSON object and NOTHING else (no prose, no markdown
fences, no <tool_call> tags):

To call tools (parallel calls allowed, 1-4 of them):
  {"tool_calls": [{"name": "web_search", "arguments": {"query": "..."}},
                   {"name": "web_fetch", "arguments": {"url": "..."}}]}

To give the final answer:
  {"final": {<the JSON object the task asks for>}}

Rules:
- You are generating TRAINING DATA for a small model that can only learn from
  evidence visible in this transcript. You may NOT use your own knowledge of
  any company or person, and you may NOT browse the web yourself. The ONLY
  admissible facts are ones in TOOL RESULT blocks above. If the answer is not
  in a tool result yet, you MUST call tools, even if you already know it.
- owner_name is the REQUIRED field. linkedin_url and title are optional:
  null them if not clearly evidenced. An honest null beats a guess.
- If a TOOL RESULT above already names the owner, STOP SEARCHING and answer.
- Never call web_fetch on linkedin.com pages (login wall, wasted call).
"""

FINAL_ONLY_INSTRUCTIONS = """\
You are the decision engine inside a web-research agent. Tools are no longer
available. Read the transcript above and reply with ONE JSON object and
NOTHING else, of the form:
  {"final": {<the JSON object the task asks for>}}
Use null for any field the gathered evidence does not support.
"""


def resolve_cli(name: str) -> str:
    for candidate in (name, f"{name}.cmd", f"{name}.exe", f"{name}.ps1"):
        found = shutil.which(candidate)
        if found and not found.endswith(".ps1"):
            return found
    raise RuntimeError(f"{name} CLI not found on PATH")


def backend_for(model: str) -> str:
    lowered = (model or "").lower()
    if "sol" in lowered or "codex" in lowered:
        return "codex"
    if "haiku" in lowered or "claude" in lowered:
        return "claude"
    raise HTTPException(400, f"unknown teacher model {model!r}: use 'sol' or 'haiku'")


def render_transcript(messages: list[dict], tools_offered: bool) -> str:
    """Flatten the OpenAI conversation into one prompt for a CLI call."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            parts.append(f"## SYSTEM\n{content}")
        elif role == "user":
            parts.append(f"## USER\n{content}")
        elif role == "assistant":
            calls = msg.get("tool_calls") or []
            if calls:
                lines = [
                    f"- {tc['function']['name']}({tc['function']['arguments']})"
                    for tc in calls
                ]
                parts.append("## ASSISTANT (called tools)\n" + "\n".join(lines))
            elif content:
                parts.append(f"## ASSISTANT\n{content}")
        elif role == "tool":
            parts.append(f"## TOOL RESULT\n{content}")
    tail = DECISION_INSTRUCTIONS if tools_offered else FINAL_ONLY_INSTRUCTIONS
    return "\n\n".join(parts) + "\n\n" + tail


def first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


async def run_codex(prompt: str) -> str:
    cli = resolve_cli("codex")
    # mkstemp holds the handle open on Windows, blocking codex from writing
    # to it, so hand codex a path inside a scratch dir instead.
    scratch = tempfile.mkdtemp(prefix="sol-cwd-")
    out_file = Path(scratch) / "last-message.txt"
    try:
        proc = await asyncio.create_subprocess_exec(
            cli, "exec", "--sandbox", "read-only", "--skip-git-repo-check",
            # Codex must not browse on its own: every fact has to arrive
            # through OUR tool calls or the trajectory carries no evidence.
            "-c", "tools.web_search=false",
            "-C", scratch, "-o", str(out_file), "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode("utf-8")), timeout=CALL_TIMEOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"codex exit {proc.returncode}: {stderr[-400:].decode(errors='replace')}")
        return out_file.read_text(encoding="utf-8", errors="replace")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


async def run_claude(prompt: str, model: str = "haiku") -> str:
    cli = resolve_cli("claude")
    proc = await asyncio.create_subprocess_exec(
        cli, "-p", "--model", model,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(prompt.encode("utf-8")), timeout=CALL_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exit {proc.returncode}: {stderr[-400:].decode(errors='replace')}")
    return stdout.decode("utf-8", errors="replace")


def to_openai_response(model: str, decision: dict | None, raw: str) -> dict:
    """Convert the teacher's decision JSON into an OpenAI chat completion."""
    message: dict[str, Any]
    finish = "stop"
    if decision and isinstance(decision.get("tool_calls"), list) and decision["tool_calls"]:
        calls = []
        for tc in decision["tool_calls"][:4]:
            name = tc.get("name")
            if name not in ("web_search", "web_fetch"):
                continue
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                },
            })
        if calls:
            message = {"role": "assistant", "content": None, "tool_calls": calls}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": raw.strip()}
    elif decision and "final" in decision:
        final = decision["final"]
        content = final if isinstance(final, str) else json.dumps(final, ensure_ascii=False, indent=2)
        message = {"role": "assistant", "content": content}
    else:
        # Unparseable teacher reply: pass it through so the service's own
        # parse handling sees it and logs it. Never invent a decision.
        message = {"role": "assistant", "content": raw.strip()}

    approx = max(1, len(raw) // 4)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": approx, "total_tokens": approx},
    }


RETRY_INVALID = (
    "\n\nYour previous reply was not ONE valid JSON object (it had a syntax "
    "error or extra text). Resend the SAME decision as one strictly valid "
    "JSON object, nothing else."
)
RETRY_NO_EVIDENCE = (
    "\n\nREJECTED: you gave a final answer but the transcript contains ZERO "
    "tool results. Facts from your own knowledge are not admissible. Reply "
    "with a tool_calls decision instead."
)


def evidence_count(messages: list[dict]) -> int:
    """Tool results present in the conversation (either protocol shape)."""
    n = sum(1 for m in messages if m.get("role") == "tool")
    n += sum(1 for m in messages
             if m.get("role") == "user" and str(m.get("content") or "").startswith("Result of "))
    return n


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(body: dict) -> JSONResponse:
    model = body.get("model") or os.environ.get("TEACHER_MODEL", "sol")
    backend = backend_for(model)
    messages = body.get("messages") or []
    tools_offered = bool(body.get("tools")) and body.get("tool_choice") != "none"
    prompt = render_transcript(messages, tools_offered)
    evidence = evidence_count(messages)

    async with LIMITS[backend]:
        started = time.time()
        raw, decision = "", None
        for attempt in range(3):
            try:
                if backend == "codex":
                    raw = await run_codex(prompt)
                else:
                    raw = await run_claude(prompt)
            except (asyncio.TimeoutError, RuntimeError) as exc:
                log.warning("teacher call failed: %s", exc)
                raise HTTPException(503, f"teacher backend error: {exc}") from exc

            decision = first_json_object(raw)
            if decision is None:
                # Sol emits sloppy JSON now and then (unclosed braces). One
                # correction round trip fixes most of them.
                log.warning("attempt %d: unparseable teacher reply, retrying", attempt + 1)
                prompt += RETRY_INVALID
                continue
            has_calls = bool(
                isinstance(decision.get("tool_calls"), list)
                and any(tc.get("name") in ("web_search", "web_fetch")
                        for tc in decision["tool_calls"] if isinstance(tc, dict))
            )
            # Anything that is not a valid tool call IS a final answer for
            # guard purposes: Sol dodges the wrapper by replying with the bare
            # answer object, and an evidence-free null is as poisonous as an
            # evidence-free name (it teaches giving up without searching).
            if not has_calls and evidence == 0 and tools_offered:
                # The poison lane: a confident teacher answering from its own
                # world knowledge produces a trajectory with no evidence, which
                # trains fabrication. Mechanically rejected, not trusted to
                # instructions.
                log.warning("attempt %d: evidence-free final rejected", attempt + 1)
                prompt += RETRY_NO_EVIDENCE
                decision = None
                continue
            break

    if decision is None and evidence == 0 and tools_offered:
        raise HTTPException(503, "teacher insists on answering without evidence")
    log.info("%s answered in %.1fs (%s)", backend, time.time() - started,
             "tool_calls" if decision and decision.get("tool_calls") else "final/other")
    return JSONResponse(to_openai_response(model, decision, raw))


@app.get("/v1/models")
@app.get("/models")
async def models() -> dict:
    return {"object": "list", "data": [
        {"id": "sol", "object": "model"},
        {"id": "haiku", "object": "model"},
    ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8944)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
