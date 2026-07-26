import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "finetune" / "build_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def trajectory(domain, *, status="success", lane="mistral-fast", lanes=None, data=None, final="Done", ts=1):
    return {
        "status": status,
        "lane": lane,
        "lanes": lanes,
        "ts": ts,
        "subject": {"company": "Example", "domain": domain},
        "data": data or {"owner_name": "Ada Smith"},
        "final_content": final,
        "messages": [
            {"role": "system", "content": "system exactly"},
            {"role": "user", "content": "find owner"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "source https://ok.test/a"},
            {"role": "assistant", "content": final},
        ],
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def build(inputs, out, **kwargs):
    return MODULE.build(
        inputs, out, min_train_rows=0, min_val_rows=0, min_train_domains=0, **kwargs
    )


def test_all_filters_and_exact_messages(tmp_path):
    rows = [
        trajectory("good.test", final="Owner https://ok.test/a"),
        trajectory("status.test", status="failed"),
        trajectory("fake.test", final="Claim https://fake.test/x"),
        trajectory("lane.test", lane=None),
        trajectory("gold.test", data={"owner_name": "Ada Jones"}),
    ]
    source = tmp_path / "input.jsonl"
    gold = tmp_path / "gold.jsonl"
    out = tmp_path / "out"
    write_jsonl(source, rows)
    write_jsonl(gold, [{"domain": "gold.test", "gold": ["smith"]}])

    stats = build([str(source)], out, val_frac=0, gold_path=gold, strip_mode=False)

    assert stats["rows_in"] == 5
    assert stats["drops"] == {
        "status": 1,
        "anti_fabrication": 1,
        "primary_url_unverified": 0,
        "lanes": 1,
        "gold": 1,
    }
    result = json.loads((out / "train.jsonl").read_text(encoding="utf-8"))
    assert result == {"messages": rows[0]["messages"]}
    assert (out / "train_domains.txt").read_text(encoding="utf-8") == "good.test\n"


def test_urls_in_nested_data_and_wildcard_lane(tmp_path):
    accepted = trajectory(
        "nested.test",
        lane=None,
        data={"owner_name": "Ada Smith", "nested": ["https://ok.test/a"]},
    )
    source = tmp_path / "nested.jsonl"
    out = tmp_path / "out"
    write_jsonl(source, [accepted])
    stats = build([str(source)], out, lane_spec="*", val_frac=1)
    assert stats["val"] == 1
    assert (out / "val_domains.txt").read_text(encoding="utf-8") == "nested.test\n"


def test_gold_fallback_and_unknown_domain(tmp_path):
    rows = [
        trajectory("gold.test", data={"answer": "Grace Smith"}),
        trajectory("unknown.test", data={"answer": "Anybody Else"}),
    ]
    source = tmp_path / "rows.jsonl"
    gold = tmp_path / "gold.jsonl"
    write_jsonl(source, rows)
    write_jsonl(gold, [{"domain": "gold.test", "gold": ["smith"]}])
    stats = build([str(source)], tmp_path / "out", val_frac=0, gold_path=gold)
    assert stats["train"] == 2


def test_domain_split_is_deterministic_and_disjoint(tmp_path):
    rows = [
        trajectory("same.test"),
        trajectory("same.test", data={"owner_name": "Bea Smith"}),
        trajectory("other.test"),
        {**trajectory("ignored.test"), "subject": "plain subject"},
    ]
    source = tmp_path / "rows.jsonl"
    write_jsonl(source, rows)
    build([str(source)], tmp_path / "one", val_frac=0.5)
    build([str(source)], tmp_path / "two", val_frac=0.5)

    for filename in ("train.jsonl", "val.jsonl", "train_domains.txt", "val_domains.txt"):
        assert (tmp_path / "one" / filename).read_bytes() == (tmp_path / "two" / filename).read_bytes()
    train_domains = set((tmp_path / "one" / "train_domains.txt").read_text().splitlines())
    val_domains = set((tmp_path / "one" / "val_domains.txt").read_text().splitlines())
    assert train_domains.isdisjoint(val_domains)
    assert ("same.test" in train_domains) != ("same.test" in val_domains)


def test_lanes_array_requires_every_entry_and_falls_back(tmp_path):
    rows = [
        trajectory("all.test", lanes=["mistral-a", "mistral-b"]),
        trajectory("mixed.test", lanes=["mistral-a", "other"]),
        trajectory("null.test", lanes=["mistral-a", None]),
        trajectory("fallback.test", lanes=[]),
    ]
    source = tmp_path / "rows.jsonl"
    write_jsonl(source, rows)
    stats = build([str(source)], tmp_path / "out", val_frac=0)
    assert stats["train"] == 2
    assert stats["drops"]["lanes"] == 2
    wildcard = build([str(source)], tmp_path / "wild", lane_spec="*", val_frac=0)
    assert wildcard["train"] == 4


def test_url_canonical_adversarial_cases(tmp_path):
    rows = [
        trajectory("query.test", final="https://ok.test/a?utm=x"),
        trajectory("punct.test", final='https://ok.test/a).'),
        trajectory("slug.test", final="https://linkedin.com/in/altered"),
        trajectory("assistant.test", final="https://only.test/a"),
    ]
    rows[2]["messages"][3]["content"] = "https://linkedin.com/in/original"
    rows[3]["messages"][2]["content"] = "Earlier https://only.test/a"
    rows[3]["messages"][3]["content"] = "no url"
    source = tmp_path / "rows.jsonl"
    write_jsonl(source, rows)
    stats = build([str(source)], tmp_path / "out", val_frac=0, strip_mode=False)
    assert stats["train"] == 2
    assert stats["drops"]["anti_fabrication"] == 2


def test_strip_mode_keeps_rows_but_removes_unverified_urls(tmp_path):
    """Default strip mode: auxiliary URLs are removed, primary linkedin drops the row."""
    aux = trajectory("aux.test", final="See https://ok.test/a and https://invented.test/x")
    linkedin_ok = trajectory(
        "person.test",
        final="Owner profile https://linkedin.com/in/real",
        data={"owner_name": "Ada Smith", "linkedin_url": "https://linkedin.com/in/real"},
    )
    linkedin_ok["messages"][3]["content"] = "https://linkedin.com/in/real"
    linkedin_fake = trajectory(
        "fakeperson.test",
        final="Owner profile https://linkedin.com/in/invented",
        data={"owner_name": "Bob Jones", "linkedin_url": "https://linkedin.com/in/invented"},
    )
    scheme = trajectory("scheme.test", final="Source http://ok.test/a")

    source = tmp_path / "rows.jsonl"
    write_jsonl(source, [aux, linkedin_ok, linkedin_fake, scheme])
    stats = build([str(source)], tmp_path / "out", lane_spec="*", val_frac=0)

    # aux kept (stripped), linkedin_ok kept, scheme kept (http==https), fake dropped
    assert stats["train"] == 3
    assert stats["drops"]["primary_url_unverified"] == 1
    assert stats["rows_url_stripped"] == 1
    text = (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8")
    assert "invented.test" not in text
    assert "ok.test/a" in text
    assert "linkedin.com/in/real" in text


def test_normalization_holdout_cap_and_manifest(tmp_path):
    rows = [
        trajectory("WWW.Keep.Test:443/path", ts=3, data={"owner_name": "C Smith"}, final="third"),
        trajectory("keep.test", ts=1, data={"owner_name": "A Smith"}, final="first"),
        trajectory("keep.test", ts=2, data={"owner_name": "B Smith"}, final="second"),
        trajectory("www.drop.test/path", ts=1),
    ]
    source = tmp_path / "rows.jsonl"
    holdout = tmp_path / "holdout.txt"
    out = tmp_path / "out"
    write_jsonl(source, rows)
    holdout.write_text("DROP.TEST:8080/path\n", encoding="utf-8")
    stats = build(
        [str(source)], out, val_frac=0, holdout_paths=[holdout], max_rows_per_domain=2
    )
    assert stats["holdout_exclusions"] == 1
    assert stats["per_domain_dedup_drops"] == 1
    assert (out / "train_domains.txt").read_text() == "keep.test\n"
    records = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert [record["messages"][-1]["content"] for record in records] == ["first", "second"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["rows_out"] == {"train": 2, "val": 0}
    assert manifest["holdout_exclusions"] == 1
    assert manifest["output_sha256"]["train.jsonl"] == MODULE.sha256_bytes(
        (out / "train.jsonl").read_bytes()
    )


def test_minimum_gate_writes_nothing(tmp_path):
    source = tmp_path / "rows.jsonl"
    out = tmp_path / "out"
    write_jsonl(source, [trajectory("one.test")])
    try:
        MODULE.build([str(source)], out, val_frac=0)
    except MODULE.DatasetMinimumError as exc:
        assert "dataset minimum gate failed" in str(exc)
    else:
        assert False, "minimum gate should fail"
    assert not out.exists()
