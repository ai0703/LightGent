import json

import pytest
import yaml

import finetune.train_modal as train_modal


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        text = " ".join(
            f"{message['role']}:{message.get('content', '')}"
            for message in messages
        )
        if tokenize:
            return list(range(len(text.split())))
        return text

    def __call__(self, text, add_special_tokens=False):
        # Token counting goes through the tokenizer call, not
        # apply_chat_template(tokenize=True), which returns a BatchEncoding
        # in transformers 5.x.
        return {"input_ids": list(range(len(str(text).split())))}


def test_dry_run_never_invokes_gpu_function_or_model_weight_loader(
    tmp_path, monkeypatch
):
    row = {
        "domain": "example.com",
        "messages": [
            {"role": "user", "content": "Find the owner"},
            {"role": "assistant", "content": "The owner is Ada"},
        ],
    }
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    contents = json.dumps(row) + "\n"
    train_path.write_text(contents, encoding="utf-8")
    val_path.write_text(contents, encoding="utf-8")

    config = {
        "model_name": train_modal.MODEL_NAME,
        "train_data": str(train_path),
        "val_data": str(val_path),
        "max_seq_len": 100,
        "max_drop_frac": 0.15,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    gpu_called = False
    weights_called = False

    def forbidden_gpu(*args, **kwargs):
        nonlocal gpu_called
        gpu_called = True
        raise AssertionError("dry_run invoked the GPU train function")

    def forbidden_weights(*args, **kwargs):
        nonlocal weights_called
        weights_called = True
        raise AssertionError("dry_run downloaded model weights")

    monkeypatch.setattr(train_modal, "train", forbidden_gpu)
    monkeypatch.setattr(train_modal, "_load_model_weights", forbidden_weights)
    monkeypatch.setattr(
        train_modal, "_load_tokenizer", lambda model_name: FakeTokenizer()
    )

    result = train_modal.dry_run(
        train_data=str(train_path),
        val_data=str(val_path),
        config_path=str(config_path),
    )

    assert result["train"]["rows"] == 1
    assert result["validation"]["dropped"] == 0
    assert gpu_called is False
    assert weights_called is False


def test_train_rejects_template_failure_before_loading_model(
    tmp_path, monkeypatch
):
    row = {
        "domain": "example.com",
        "messages": [
            {"role": "user", "content": "Find the owner"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": {"query": "example.com owner"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "Search result"},
        ],
    }
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = {
        "model_name": train_modal.MODEL_NAME,
        "train_data": str(train_path),
        "val_data": str(train_path),
        "max_seq_len": 100,
        "max_drop_frac": 0.15,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    weights_called = False

    def forbidden_weights(*args, **kwargs):
        nonlocal weights_called
        weights_called = True
        raise AssertionError("train downloaded model weights")

    monkeypatch.setattr(
        train_modal, "CONFIG_REMOTE_PATH", str(config_path)
    )
    monkeypatch.setattr(train_modal, "_load_model_weights", forbidden_weights)
    monkeypatch.setattr(
        train_modal, "_load_tokenizer", lambda model_name: FakeTokenizer()
    )

    with pytest.raises(ValueError, match="chat-template validation failed"):
        train_modal.train.local()

    assert weights_called is False
