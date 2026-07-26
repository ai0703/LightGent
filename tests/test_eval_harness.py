import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx

import finetune.eval_harness as harness
import lightgent_service
from finetune.build_dataset import canonical_urls as dataset_canonical_urls
from lightgent_service import ResearchRequest, run_agent


def message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def response(msg):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        model="mock-model",
        model_extra={},
    )


class MockAsyncOpenAI:
    created = []

    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )
        self.created.append((base_url, api_key))

    async def create(self, **kwargs):
        messages = kwargs["messages"]
        if messages and messages[0].get("role") == "system":
            prompt = messages[0]["content"]
            if "negative.example" in prompt:
                data = {
                    "owner_name": "Nobody",
                    "title": None,
                    "linkedin_url": "https://linkedin.com/in/planted-fake",
                }
            elif "berg.example" in prompt:
                data = {
                    "owner_name": "J. van der Berg",
                    "title": "Owner",
                    "linkedin_url": "https://nl.linkedin.com/in/jan-berg",
                }
            else:
                data = {"owner_name": "Alice Example", "title": "Owner",
                        "linkedin_url": None}
            return response(message(json.dumps(data)))
        if any(item.get("role") == "tool" for item in messages):
            return response(message("The CEO is Dario Amodei."))
        call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="web_search", arguments='{"query":"Anthropic CEO"}'
            ),
        )
        return response(message(None, [call]))


def write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_end_to_end_report_fabrication_tussenvoegsel_and_adjudication(
    tmp_path, monkeypatch,
):
    data = tmp_path / "data"
    data.mkdir()
    (data / "dev_domains.txt").write_text("berg.example\n", encoding="utf-8")
    (data / "test_domains.txt").write_text("test.example\n", encoding="utf-8")
    train = data / "train_domains.txt"
    train.write_text("train.example\n", encoding="utf-8")
    negatives = tmp_path / "negatives.json"
    negatives.write_text(json.dumps([{
        "company": "Negative Co", "domain": "negative.example",
        "found_name": "Wrong Person", "found_title": "CEO",
        "found_linkedin": "https://linkedin.com/in/wrong",
        "gt_lastnames": ["real"],
    }]), encoding="utf-8")
    overnight = tmp_path / "overnight.jsonl"
    write_jsonl(overnight, [{
        "company": "Berg BV", "domain": "berg.example", "status": "success",
        "got": "Jan van der Berg", "title": "Eigenaar",
        "linkedin": "https://linkedin.com/in/jan-van-der-berg",
    }])
    monkeypatch.setattr(harness, "DATA_DIR", data)
    monkeypatch.setattr(harness, "NEGATIVES_PATH", negatives)
    monkeypatch.setattr(harness, "OVERNIGHT_PATH", overnight)
    monkeypatch.setattr(lightgent_service, "AsyncOpenAI", MockAsyncOpenAI)
    monkeypatch.setattr(harness, "AsyncOpenAI", MockAsyncOpenAI)
    out = tmp_path / "report.json"
    args = argparse.Namespace(
        base_url="http://mock/v1", model="mock", api_key="key", out=out,
        limit=None, split="dev", train_domains=train, sets="a,b",
        concurrency=2,
    )

    report = asyncio.run(harness.evaluate(args))
    out.write_text(json.dumps(report), encoding="utf-8")

    assert set(report) >= {"configuration", "sets", "probes", "adjudication_path"}
    assert set(report["sets"]) == {"a", "b"}
    assert report["sets"]["a"]["fabricated_url_count"] == 1
    assert report["sets"]["a"]["rows"][0]["fabricated_urls"] == [
        "linkedin.com/in/planted-fake"
    ]
    surname = report["sets"]["b"]["metrics"]["surname_agreement"]
    assert surname["count"] == 1
    assert surname["total"] == 1
    assert len(surname["ci95"]) == 2
    adjudication = [
        json.loads(line)
        for line in Path(report["adjudication_path"]).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(adjudication) == 1
    assert adjudication[0]["raw_agreement_flags"]["surname_agreement"] is True
    assert report["probes"]["native_tool_call"]["pass"] is True
    assert report["probes"]["two_turn_loop"]["pass"] is True


def test_run_agent_overrides_do_not_leak_between_configurations(
    tmp_path, monkeypatch,
):
    MockAsyncOpenAI.created.clear()
    monkeypatch.setattr(lightgent_service, "AsyncOpenAI", MockAsyncOpenAI)
    original = (
        lightgent_service.settings.llm_base_url,
        lightgent_service.settings.llm_model,
        lightgent_service.settings.llm_api_key,
        lightgent_service.settings.trajectory_log_dir,
    )
    request = ResearchRequest(
        task="Find owner", context={"domain": "one.example"},
        output_fields={"owner_name": "name"},
    )

    async def exercise():
        async with httpx.AsyncClient() as http:
            await run_agent(
                request, http, base_url="http://first/v1", model="first",
                api_key="first-key", trajectory_dir=tmp_path / "first",
            )
            await run_agent(
                request, http, base_url="http://second/v1", model="second",
                api_key="second-key", trajectory_dir=tmp_path / "second",
            )

    asyncio.run(exercise())

    assert MockAsyncOpenAI.created == [
        ("http://first/v1", "first-key"),
        ("http://second/v1", "second-key"),
    ]
    first = json.loads(next((tmp_path / "first").glob("*.jsonl")).read_text(
        encoding="utf-8"
    ))
    second = json.loads(next((tmp_path / "second").glob("*.jsonl")).read_text(
        encoding="utf-8"
    ))
    assert first["model"] == "first"
    assert second["model"] == "second"
    assert original == (
        lightgent_service.settings.llm_base_url,
        lightgent_service.settings.llm_model,
        lightgent_service.settings.llm_api_key,
        lightgent_service.settings.trajectory_log_dir,
    )


def test_domain_disjointness_and_canonical_url_rule(tmp_path):
    chosen = tmp_path / "dev.txt"
    other = tmp_path / "test.txt"
    train = tmp_path / "train.txt"
    chosen.write_text("https://WWW.Example.com:443/path\n", encoding="utf-8")
    other.write_text("other.example\n", encoding="utf-8")
    train.write_text("example.com\n", encoding="utf-8")
    try:
        harness.assert_disjoint(chosen, other, train)
    except ValueError as exc:
        assert "pairwise disjoint" in str(exc)
    else:
        raise AssertionError("overlap was not detected")

    vector = "See HTTPS://Example.COM/a/ and http://x.test:8080/p)."
    assert harness.canonical_urls(vector) == dataset_canonical_urls(vector)
    assert harness.canonical_urls(vector) == {
        "example.com/a", "x.test/p"
    }
