# Qwen3 4B QLoRA fine-tuning on Modal

This Modal app fine-tunes `Qwen/Qwen3-4B-Instruct-2507` with QLoRA,
merges the adapter into an fp16 model, and runs a one-request tool-call
smoke test against the merged artifact.

The training configuration is in `finetune/train_config.yaml`. Modal copies
that file into the immutable training image at build time. Commit any config
changes before treating a run as reproducible. The data files must be JSONL,
with one `{"messages": [...]}` object per line.

Training is blocked until the CPU-only preflight passes on a representative
dataset row containing an assistant `tool_calls` turn and a `role: "tool"`
result. Preflight uses the exact Qwen3 4B tokenizer and reports the row's true
chat-template token length. The full dry-run must then tokenize every train and
validation row before approval to launch training.

## Launch gate and cost

Check the available Modal credit balance before launching. The launch decision
requires this credit check first.

An A100 40 GB GPU on Modal costs roughly USD 2.10 to USD 2.50 per hour. A run
is expected to take 2 to 4 hours, so budget roughly USD 5 to USD 10 per run.
Modal starter credits are expected to cover this. The 4 hour function timeout
is the hard cost cap.

The app requests `A100-40GB` for training and `L40S` for smoke testing. If a
Modal workspace rejects either GPU alias, change the decorator GPU string in
`train_modal.py` to `A100` for training or `A10` for smoke testing.

Only one training run may be active at a time. Retries are disabled. Every
rerun, including a rerun after failure, timeout, or cancellation, requires
fresh approval and a new credit and price check.

## Commands

Run all commands from the repository root. The examples use `python -m modal`
because the bare `modal` command is not on PATH on the target machine.

Create the volume once:

```powershell
python -m modal volume create lightgent-finetune
```

Upload the training and validation data:

```powershell
python -m modal volume put lightgent-finetune .\path\to\train.jsonl data/train.jsonl
python -m modal volume put lightgent-finetune .\path\to\val.jsonl data/val.jsonl
```

Install the lightweight local validation dependencies, then run preflight on
the local training JSONL. This downloads tokenizer files but no model weights:

```powershell
python -m pip install transformers==4.53.2 PyYAML==6.0.2
python -m modal run finetune/train_modal.py::preflight --data-path .\path\to\train.jsonl
```

Run the separate CPU-only local dry-run. Local path overrides keep it away
from the Modal volume paths in the checked-in configuration:

```powershell
python -m modal run finetune/train_modal.py::dry_run --train-data .\path\to\train.jsonl --val-data .\path\to\val.jsonl
```

Start training and merge the adapter. Replace the run name consistently in
later commands:

```powershell
python -m modal run finetune/train_modal.py::train --run-name qwen3-4b-lightgent
```

Download the merged model:

```powershell
python -m modal volume get lightgent-finetune merged/qwen3-4b-lightgent .\artifacts\qwen3-4b-lightgent
```

Run the load-and-emit smoke test against the stock base model before training:

```powershell
python -m modal run finetune/train_modal.py::serve_smoke --model Qwen/Qwen3-4B-Instruct-2507
```

After training, run the same smoke test against the merged volume model:

```powershell
python -m modal run finetune/train_modal.py::serve_smoke --model /vol/merged/qwen3-4b-lightgent
```

The smoke function does not start a server. It loads the merged model, requests
one `web_search` tool call, validates its structure, prints it, and exits.

## Required cleanup

After the merged model is downloaded and verified, remove uploaded data and
checkpoints from the private volume:

```powershell
python -m modal volume rm lightgent-finetune data -r
python -m modal volume rm lightgent-finetune checkpoints -r
python -m modal volume ls lightgent-finetune
```

Perform the same cleanup after every failed, timed-out, or cancelled run. The
final listing is the retention record showing that `/vol/data` and
`/vol/checkpoints` are gone.
