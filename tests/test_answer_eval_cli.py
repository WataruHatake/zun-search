import copy
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_app.answer_eval import capture_answers, main
from rag_app.answer_evaluation import evaluate_answers

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/answer_evaluation_demo.json"
PREDICTIONS = ROOT / "data/answer_predictions_mock.json"


def demo():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def test_offline_score_needs_only_standard_library(tmp_path):
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "rag_app.answer_eval",
            "score",
            "--dataset",
            str(DATASET),
            "--predictions",
            str(PREDICTIONS),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["run_type"] == "mock"
    assert report["summary"]["cases"] == 11
    assert report["summary"]["human"]["correctness"]["mean_score"] is None
    rows = {row["case_id"]: row for row in report["records"]}
    assert not rows["uncited_extra_claim"]["automatic"]["citation_validation"]["valid"]
    assert not rows["missing_middle_citation"]["automatic"]["citation_validation"]["valid"]
    for case_id in ("cited_unsupported_claim", "contradicted_number", "irrelevant_citation"):
        assert rows[case_id]["automatic"]["citation_validation"]["valid"]
        assert rows[case_id]["human_review_required"]


def test_empty_review_round_trip_and_input_protection(tmp_path):
    review = tmp_path / "review.json"
    output = tmp_path / "report.json"
    common = ["--dataset", str(DATASET), "--predictions", str(PREDICTIONS)]
    main(["review-template", *common, "--output", str(review)])
    main(["score", *common, "--reviews", str(review), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["human"]["reviewed_cases"] == 0
    with pytest.raises(SystemExit) as exc:
        main(["score", *common, "--output", str(DATASET)])
    assert exc.value.code == 2


class FakeIndex:
    def files(self):
        return [
            {
                "id": 7,
                "relpath": "project.md",
                "path": str(ROOT / "data/answer_evaluation_sources/project.md"),
            }
        ]


def one_case():
    dataset = demo()
    dataset["cases"] = dataset["cases"][:1]
    return dataset


def test_capture_preserves_failed_raw_answer_and_delivered_refusal():
    dataset = one_case()
    raw = {"supported": True, "answer": "費用は120万円です[1]。別の無根拠な主張。", "citations": []}
    delivered = {
        "status": "validation_failed",
        "answer": "根拠を確認できません。",
        "citations": [],
        "raw_response": raw,
        "retrieved": dataset["cases"][0]["evidence"],
    }
    with patch("rag_app.service.answer", return_value=delivered):
        predictions = capture_answers(FakeIndex(), dataset, object(), {}, run_type="mock")
    record = predictions["records"][0]
    assert predictions["run_type"] == "mock"
    assert record["response"] == raw
    assert record["delivered"]["answer"] != raw["answer"]
    assert record["capture_status"] == "validation_failed"
    assert not evaluate_answers(dataset, predictions)["records"][0]["automatic"]["citation_validation"][
        "valid"
    ]


@pytest.mark.parametrize("failure", [None, RuntimeError("model offline")])
def test_no_generation_is_not_successful_abstention(failure):
    dataset = one_case()
    delivered = {
        "status": "no_evidence",
        "answer": "本文がありません。",
        "raw_response": None,
        "retrieved": [],
    }
    with patch("rag_app.service.answer", return_value=delivered, side_effect=failure):
        predictions = capture_answers(FakeIndex(), dataset, object(), {}, run_type="mock")
    record = predictions["records"][0]
    assert record["generation_attempted"] is False
    assert record["response"] == {}
    report = evaluate_answers(dataset, predictions)
    assert report["summary"]["automatic"]["declared_answerability_denominator"] == 0
    assert report["summary"]["human"]["abstention"]["mean_score"] is None


def test_capture_rejects_changed_gold_source_before_generation():
    dataset = copy.deepcopy(one_case())
    dataset["cases"][0]["source_sha256"] = "0" * 64
    with patch("rag_app.service.answer") as answer:
        with pytest.raises(ValueError, match="評価資料が変更"):
            capture_answers(FakeIndex(), dataset, object(), {}, run_type="mock")
        answer.assert_not_called()


def test_capture_rejects_missing_source():
    dataset = one_case()
    dataset["cases"][0]["source"] = "missing.md"
    with pytest.raises(ValueError, match="索引にない"):
        capture_answers(FakeIndex(), dataset, object(), {}, run_type="mock")
