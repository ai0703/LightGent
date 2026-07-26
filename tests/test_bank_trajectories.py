import argparse
import asyncio
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from finetune import bank_trajectories
from lightgent_service import ResearchResponse


def write_csv(path: Path) -> None:
    rows = [
        ("Holdout Co", "https://www.holdout.example/path", "A", "NL"),
        ("Done Co", "done.example:443", "B", "NL"),
        ("First Co", "www.first.example", "C", "NL"),
        ("Duplicate First", "FIRST.EXAMPLE/path", "D", "NL"),
        ("Second Co", "second.example", "E", "NL"),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["company_name", "domain", "city", "country"])
        writer.writerows(rows)


def test_skips_holdout_resume_and_stops_at_target() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        tmp_path = Path(directory)
        input_path = tmp_path / "companies.csv"
        holdout = tmp_path / "holdout.txt"
        state = tmp_path / "state" / "results.jsonl"
        write_csv(input_path)
        holdout.write_text("HOLDOUT.EXAMPLE\n", encoding="utf-8")
        state.parent.mkdir()
        state.write_text(
            json.dumps({"domain": "www.done.example", "status": "success"}) + "\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            input=str(input_path),
            holdout=[str(holdout)],
            state=str(state),
            trajectory_dir=str(tmp_path / "trajectories"),
            concurrency=1,
            target_successes=2,
            limit=0,
            base_url=None,
            model=None,
            api_key=None,
        )
        mocked = AsyncMock(
            return_value=ResearchResponse(
                status="success", data={}, iterations=3, tool_calls=4
            )
        )

        with patch.object(bank_trajectories, "run_agent", mocked):
            summary = asyncio.run(bank_trajectories.bank(args))

        lines = [
            json.loads(line)
            for line in state.read_text(encoding="utf-8").splitlines()
        ]
        assert mocked.await_count == 1
        assert mocked.await_args.args[0].context["domain"] == "first.example"
        assert mocked.await_args.kwargs["trajectory_dir"] == Path(args.trajectory_dir)
        assert len(lines) == 2
        assert lines[-1]["domain"] == "first.example"
        assert summary["attempted"] == 1
        assert summary["successes"] == 2
        assert summary["stopped"] == "target"
