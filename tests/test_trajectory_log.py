import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

import lightgent_service as service


def completion(content=None, tool_calls=None, lane=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    extra = {"_broker": {"lane": lane}} if lane is not None else {}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        model_extra=extra,
    )


def tool_call(call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="web_search",
            arguments='{"query":"LightGent"}',
        ),
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = responses

    async def create(self, **kwargs):
        return self.responses.pop(0)


class FakeAsyncOpenAI:
    responses = []

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(list(self.responses))
        )


@pytest.fixture
def research_request():
    return service.ResearchRequest(
        task="Find the answer",
        context={"company": "Example"},
        output_fields={"answer": "the answer"},
    )


def test_setting_unset_creates_no_file(tmp_path, monkeypatch, research_request):
    monkeypatch.setattr(service.settings, "trajectory_log_dir", "")
    monkeypatch.setattr(service, "AsyncOpenAI", FakeAsyncOpenAI)
    FakeAsyncOpenAI.responses = [completion('{"answer":"ok"}')]

    result = asyncio.run(_run(research_request))

    assert result.status == "success"
    assert list(tmp_path.iterdir()) == []


def test_setting_set_writes_exactly_one_json_line(
        tmp_path, monkeypatch, research_request):
    monkeypatch.setattr(service.settings, "trajectory_log_dir", str(tmp_path))
    monkeypatch.setattr(service, "AsyncOpenAI", FakeAsyncOpenAI)
    FakeAsyncOpenAI.responses = [completion('{"answer":"ok"}', lane="lane-a")]

    asyncio.run(_run(research_request))

    files = list(tmp_path.glob("trajectories-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "success"


def test_full_messages_lane_and_status_are_logged(
        tmp_path, monkeypatch, research_request):
    monkeypatch.setattr(service.settings, "trajectory_log_dir", str(tmp_path))
    monkeypatch.setattr(service, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(service, "_dispatch", _fake_dispatch)
    FakeAsyncOpenAI.responses = [
        completion(tool_calls=[tool_call()], lane="lane-first"),
        completion('{"answer":"found"}', lane="lane-last"),
    ]

    result = asyncio.run(_run(research_request))

    path = next(tmp_path.glob("trajectories-*.jsonl"))
    record = json.loads(path.read_text(encoding="utf-8"))
    messages = record["messages"]

    assert result.status == "success"
    assert messages[0]["role"] == "system"
    assert any(m["role"] == "assistant" and m.get("tool_calls") for m in messages)
    assert any(m["role"] == "tool" and m["content"] == "tool result" for m in messages)
    assert messages[-1] == {"role": "assistant", "content": '{"answer":"found"}'}
    assert record["lane"] == "lane-last"
    assert record["lanes"] == ["lane-first", "lane-last"]
    assert record["status"] == "success"
    assert record["final_content"] == '{"answer":"found"}'
    assert record["tool_calls"] == 1


def test_lanes_include_null_and_lane_is_last_non_null(
        tmp_path, monkeypatch, research_request):
    monkeypatch.setattr(service.settings, "trajectory_log_dir", str(tmp_path))
    monkeypatch.setattr(service, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(service, "_dispatch", _fake_dispatch)
    FakeAsyncOpenAI.responses = [
        completion(tool_calls=[tool_call()], lane="lane-first"),
        completion('{"answer":"found"}'),
    ]

    asyncio.run(_run(research_request))

    record = json.loads(next(tmp_path.glob("trajectories-*.jsonl")).read_text(
        encoding="utf-8"))
    assert record["lanes"] == ["lane-first", None]
    assert record["lane"] == "lane-first"


@pytest.mark.parametrize(
    ("responses", "max_iterations"),
    [
        ([completion("not json")], None),
        ([completion(tool_calls=[tool_call()])], 1),
    ],
    ids=["parse_error", "max_iterations"],
)
def test_parse_error_outcomes_log_one_line(
        tmp_path, monkeypatch, research_request, responses, max_iterations):
    monkeypatch.setattr(service.settings, "trajectory_log_dir", str(tmp_path))
    monkeypatch.setattr(service, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(service, "_dispatch", _fake_dispatch)
    FakeAsyncOpenAI.responses = responses
    research_request.max_iterations = max_iterations

    result = asyncio.run(_run(research_request))

    lines = next(tmp_path.glob("trajectories-*.jsonl")).read_text(
        encoding="utf-8").splitlines()
    assert result.status == "parse_error"
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "parse_error"


def test_concurrent_runs_append_well_formed_lines(
        tmp_path, monkeypatch, research_request):
    run_count = 20
    monkeypatch.setattr(service.settings, "trajectory_log_dir", str(tmp_path))
    monkeypatch.setattr(service, "AsyncOpenAI", FakeAsyncOpenAI)
    FakeAsyncOpenAI.responses = [completion('{"answer":"ok"}')]

    async def run_all():
        async with httpx.AsyncClient() as http:
            await asyncio.gather(*[
                service.run_agent(research_request, http)
                for _ in range(run_count)
            ])

    asyncio.run(run_all())

    lines = next(tmp_path.glob("trajectories-*.jsonl")).read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == run_count
    assert all(json.loads(line)["status"] == "success" for line in lines)


class FailingHandle:
    def __init__(self, failure):
        self.failure = failure

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def write(self, value):
        if self.failure == "write":
            raise OSError("write failed")
        return len(value)

    def flush(self):
        if self.failure == "flush":
            raise OSError("flush failed")

    def fileno(self):
        return 123


@pytest.mark.parametrize(
    "failure", ["mkdir", "open", "write", "flush", "fsync"])
def test_file_operation_failures_warn_without_raising(
        tmp_path, monkeypatch, caplog, failure):
    monkeypatch.setattr(service.settings, "trajectory_log_dir", str(tmp_path))

    if failure == "mkdir":
        def fail_mkdir(*args, **kwargs):
            raise OSError("mkdir failed")
        monkeypatch.setattr(service.Path, "mkdir", fail_mkdir)
    elif failure == "open":
        def fail_open(*args, **kwargs):
            raise OSError("open failed")
        monkeypatch.setattr(service.Path, "open", fail_open)
    else:
        monkeypatch.setattr(
            service.Path, "open",
            lambda *args, **kwargs: FailingHandle(failure))
        if failure == "fsync":
            def fail_fsync(fd):
                raise OSError("fsync failed")
            monkeypatch.setattr(service.os, "fsync", fail_fsync)

    with caplog.at_level(logging.WARNING, logger="lightgent"):
        asyncio.run(service._write_trajectory({"status": "success"}))

    assert "failed to write trajectory log" in caplog.text


async def _fake_dispatch(tool_call_data, http):
    return "tool result"


async def _run(research_request):
    async with httpx.AsyncClient() as http:
        return await service.run_agent(research_request, http)
