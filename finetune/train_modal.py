"""Modal QLoRA training and merged-model smoke test for Lightgent."""

from collections import Counter
import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "lightgent-finetune"
VOLUME_NAME = "lightgent-finetune"
CONFIG_REMOTE_PATH = "/opt/lightgent/train_config.yaml"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_config_path = Path(__file__).with_name("train_config.yaml")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.53.2",
        "trl==0.19.1",
        "peft==0.16.0",
        "bitsandbytes==0.46.1",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        "PyYAML==6.0.2",
    )
    .add_local_file(str(_config_path), CONFIG_REMOTE_PATH, copy=True)
)

smoke_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm==0.8.5.post1",
    "transformers==4.53.2",
)


def _valid_run_name(run_name: str) -> str:
    """Reject path traversal while permitting useful run labels."""
    if not run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be non-empty")
    if Path(run_name).name != run_name or "/" in run_name or "\\" in run_name:
        raise ValueError("run_name must be a single path component")
    return run_name


def _load_config(config_path: str) -> dict[str, Any]:
    import yaml

    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if config.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"model_name must be the exact training model {MODEL_NAME!r}"
        )
    max_drop_frac = config.get("max_drop_frac")
    if not isinstance(max_drop_frac, (int, float)) or not 0 <= max_drop_frac <= 1:
        raise ValueError("max_drop_frac must be a number between 0 and 1")
    return config


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _read_rows(path: str, split_name: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                messages = row["messages"]
                if not isinstance(messages, list):
                    raise TypeError("messages must be a list")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid {split_name} row at {path}:{line_number}"
                ) from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in the {split_name} split")
    return rows


def _row_domain(row: dict[str, Any]) -> str:
    for container in (row, row.get("metadata"), row.get("trajectory")):
        if isinstance(container, dict):
            for key in ("domain", "company_domain", "website_domain"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return "<unknown>"


def _render_and_count(tokenizer, messages: list[dict[str, Any]]) -> tuple[str, int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Count from the rendered text. transformers 5.x returns a BatchEncoding
    # from apply_chat_template(tokenize=True), so len() there gives the number
    # of dict keys (2), not the number of tokens.
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return text, len(token_ids)


def _validate_chat_template_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    split_name: str,
) -> list[int]:
    """Validate the first and token-longest rows as real tool trajectories."""
    rendered_rows = []
    for index, row in enumerate(rows):
        try:
            rendered, token_count = _render_and_count(
                tokenizer, row["messages"]
            )
        except Exception as exc:
            raise ValueError(
                f"{split_name} chat-template validation failed for row "
                f"{index + 1}: {exc}"
            ) from exc
        rendered_rows.append((rendered, token_count))

    longest_index = max(
        range(len(rendered_rows)),
        key=lambda index: rendered_rows[index][1],
    )
    selected_indexes = list(dict.fromkeys((0, longest_index)))
    for index in selected_indexes:
        messages = rows[index]["messages"]
        rendered = rendered_rows[index][0]
        has_tool_result = any(
            message.get("role") == "tool" for message in messages
        )
        tool_call_messages = [
            message
            for message in messages
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        if not tool_call_messages or not has_tool_result:
            raise ValueError(
                f"{split_name} chat-template validation failed for row "
                f"{index + 1}: assistant tool_calls and tool results are required"
            )
        for message in tool_call_messages:
            for call in message["tool_calls"]:
                function = call.get("function", {})
                name = function.get("name")
                if not name or name not in rendered:
                    raise ValueError(
                        f"{split_name} chat-template validation failed for row "
                        f"{index + 1}: assistant tool call {name!r} is absent "
                        "from rendered text"
                    )
    return selected_indexes


def _validate_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    max_seq_len: int,
    split_name: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    kept = []
    token_lengths = []
    dropped_by_domain: Counter[str] = Counter()
    for row in rows:
        text, token_count = _render_and_count(tokenizer, row["messages"])
        token_lengths.append(token_count)
        if token_count > max_seq_len:
            dropped_by_domain[_row_domain(row)] += 1
        else:
            kept.append({"text": text})

    dropped = sum(dropped_by_domain.values())
    stats = {
        "split": split_name,
        "rows": len(rows),
        "kept": len(kept),
        "dropped": dropped,
        "drop_frac": dropped / len(rows),
        "min_tokens": min(token_lengths),
        "max_tokens": max(token_lengths),
        "mean_tokens": sum(token_lengths) / len(token_lengths),
        "dropped_by_domain": dict(sorted(dropped_by_domain.items())),
    }
    print(
        f"{split_name}: rows={stats['rows']} kept={stats['kept']} "
        f"dropped={dropped} tokens(min/mean/max)="
        f"{stats['min_tokens']}/{stats['mean_tokens']:.1f}/"
        f"{stats['max_tokens']}"
    )
    if dropped_by_domain:
        print(
            f"{split_name}: overlong rows by domain: "
            f"{dict(sorted(dropped_by_domain.items()))}"
        )
    return kept, stats


def _checked_split(
    path: str,
    split_name: str,
    tokenizer,
    config: dict[str, Any],
    enforce_drop_frac: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = _read_rows(path, split_name)
    kept, stats = _validate_rows(
        rows, tokenizer, config["max_seq_len"], split_name
    )
    over_limit = stats["drop_frac"] > config["max_drop_frac"]
    if over_limit and enforce_drop_frac:
        raise ValueError(
            f"{split_name} overlong drop rate {stats['drop_frac']:.2%} exceeds "
            f"max_drop_frac {config['max_drop_frac']:.2%}"
        )
    if over_limit:
        # Validation only measures progress, it does not shape the weights, and
        # on a small split a couple of long rows trip a percentage threshold
        # without indicating a data problem. Warn instead of aborting.
        print(
            f"WARNING: {split_name} overlong drop rate {stats['drop_frac']:.2%} "
            f"exceeds max_drop_frac {config['max_drop_frac']:.2%}; continuing "
            f"with {len(kept)} rows because this split is not trained on"
        )
    if not kept:
        raise ValueError(f"No usable rows remain in the {split_name} split")
    return kept, stats


def _load_model_weights(config, bnb_config, torch):
    """The only base-model weight loading path used by train()."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        use_cache=False,
    )


@app.local_entrypoint()
def preflight(
    data_path: str,
    config_path: str = str(_config_path),
):
    """Validate one real tool trajectory with the exact training tokenizer."""
    config = _load_config(config_path)
    tokenizer = _load_tokenizer(config["model_name"])
    rows = _read_rows(data_path, "preflight")
    representative = None
    for row in rows:
        messages = row["messages"]
        has_tool_call = any(
            message.get("role") == "assistant" and message.get("tool_calls")
            for message in messages
        )
        has_tool_result = any(
            message.get("role") == "tool" for message in messages
        )
        if has_tool_call and has_tool_result:
            representative = row
            break
    if representative is None:
        raise ValueError(
            "Preflight requires a row with assistant tool_calls and tool results"
        )

    _validate_chat_template_rows([representative], tokenizer, "Preflight")
    _, token_count = _render_and_count(tokenizer, representative["messages"])
    print(
        f"Preflight passed with {MODEL_NAME}; true token length={token_count}"
    )
    return {"model": MODEL_NAME, "token_length": token_count}


@app.local_entrypoint()
def dry_run(
    train_data: str = "",
    val_data: str = "",
    config_path: str = str(_config_path),
):
    """CPU-only full-dataset tokenization and length validation."""
    config = _load_config(config_path)
    tokenizer = _load_tokenizer(config["model_name"])
    paths = {
        "train": train_data or config["train_data"],
        "validation": val_data or config["val_data"],
    }
    results = {}
    for split_name, path in paths.items():
        rows = _read_rows(path, split_name)
        _, results[split_name] = _validate_rows(
            rows, tokenizer, config["max_seq_len"], split_name
        )
    print(
        "Dry-run complete; no GPU or model weights were requested. "
        f"overlong-drop count={sum(s['dropped'] for s in results.values())}"
    )
    return results


@app.function(
    image=train_image,
    # 40GB OOMs on these 16k-token trajectories (attention activations alone
    # asked for 9 GiB). 80GB clears it for about 0.40 USD/hr more.
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    volumes={"/vol": volume},
    retries=0,
)
def train(run_name: str = "qwen3-4b-lightgent"):
    """QLoRA-fine-tune, then merge and persist fp16 weights."""
    run_name = _valid_run_name(run_name)
    config = _load_config(CONFIG_REMOTE_PATH)
    tokenizer = _load_tokenizer(config["model_name"])
    raw_train_rows = _read_rows(config["train_data"], "train")
    checked_indexes = _validate_chat_template_rows(
        raw_train_rows, tokenizer, "Training"
    )
    print(
        "Training chat-template validation passed for rows "
        f"{[index + 1 for index in checked_indexes]}"
    )

    import gc
    import os

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    train_rows, _ = _checked_split(
        config["train_data"], "train", tokenizer, config
    )
    val_rows, _ = _checked_split(
        config["val_data"], "validation", tokenizer, config, enforce_drop_frac=False
    )
    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)

    quantization = config["quantization"]
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[quantization["bnb_4bit_compute_dtype"]]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization["load_in_4bit"],
        bnb_4bit_quant_type=quantization["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quantization[
            "bnb_4bit_use_double_quant"
        ],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = _load_model_weights(config, bnb_config, torch)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config["gradient_checkpointing"],
    )

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    checkpoint_dir = f"/vol/checkpoints/{run_name}"
    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        max_length=config["max_seq_len"],
        packing=config["packing"],
        dataset_text_field="text",
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        eval_strategy=config["eval_strategy"],
        save_total_limit=config["save_total_limit"],
        gradient_checkpointing=config["gradient_checkpointing"],
        bf16=config["bf16"],
        fp16=config["fp16"],
        seed=config["seed"],
        report_to="none",
        # Paged 8-bit optimiser states plus non-reentrant checkpointing keep
        # 16k-token sequences inside GPU memory.
        optim=config.get("optim", "paged_adamw_8bit"),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()

    adapter_dir = os.path.join(checkpoint_dir, "final-adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    volume.commit()

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(
        base_model, adapter_dir
    ).merge_and_unload()
    merged_dir = f"/vol/merged/{run_name}"
    merged_model.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(merged_dir)
    volume.commit()

    download_command = (
        "python -m modal volume get lightgent-finetune "
        f"merged/{run_name} <local-dir>"
    )
    print(f"Merged fp16 model saved to {merged_dir}")
    print(download_command)
    return merged_dir


@app.function(
    image=smoke_image,
    gpu="L40S",
    timeout=15 * 60,
    volumes={"/vol": volume},
    retries=0,
)
def serve_smoke(model: str = MODEL_NAME):
    """Load a volume model or HuggingFace model, emit a tool call, and exit."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager

    if model.startswith("/vol/merged/"):
        run_name = model.removeprefix("/vol/merged/")
        _valid_run_name(run_name)
        model_path = model
    elif model and not model.startswith("/"):
        model_path = model
    else:
        raise ValueError(
            "model must be /vol/merged/{run_name} or a HuggingFace model id"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "Use the provided tool. Do not answer from memory.",
        },
        {
            "role": "user",
            "content": "What is the current weather in Lagos, Nigeria?",
        },
    ]
    request = ChatCompletionRequest(
        model=model_path,
        messages=messages,
        tools=tools,
        tool_choice="required",
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    result = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=256),
    )[0].outputs[0].text

    parser_class = ToolParserManager.get_tool_parser("hermes")
    if parser_class is None:
        raise AssertionError("vLLM did not register the hermes tool parser")
    parsed = parser_class(tokenizer).extract_tool_calls(result, request)
    tool_calls = parsed.tool_calls if parsed.tools_called else []
    assert tool_calls, f"Expected a tool call, got: {result!r}"
    call = tool_calls[0]
    assert call.function.name == "web_search", call
    arguments = json.loads(call.function.arguments)
    assert isinstance(arguments.get("query"), str) and arguments["query"], arguments

    printable = {
        "name": call.function.name,
        "arguments": arguments,
        "raw_output": result,
    }
    print(json.dumps(printable, indent=2))
    return printable
