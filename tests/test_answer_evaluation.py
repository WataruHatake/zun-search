import copy
import json

import pytest

from rag_app.answer_evaluation import (
    content_sha256,
    evaluate_answers,
    review_template,
    validate_answer_dataset,
)


@pytest.fixture
def inputs():
    evidence = [{"id": 1, "location": "1行", "text": "担当者は田中です。予算は120万円です。"}]
    case = {
        "id": "supported",
        "question": "担当者は誰ですか？",
        "source": "synthetic.txt",
        "answerable": True,
        "reference_answer": "担当者は田中です。",
        "required_claims": [
            {
                "id": "owner",
                "text": "担当者は田中",
                "evidence": [{"source_id": 1, "quote": "担当者は田中です。"}],
            }
        ],
        "forbidden_claims": ["担当者は佐藤", "納期は明日"],
        "evidence": evidence,
    }
    dataset = {
        "schema_version": 1,
        "kind": "answer_grounding",
        "provenance": {"type": "synthetic", "description": "テスト専用の架空資料"},
        "cases": [
            case,
            {**copy.deepcopy(case), "id": "unsupported"},
            {
                **copy.deepcopy(case),
                "id": "unanswerable",
                "question": "納期はいつですか？",
                "answerable": False,
                "reference_answer": "資料に納期の記載がありません。",
                "required_claims": [],
            },
        ],
    }
    predictions = {
        "schema_version": 1,
        "kind": "answer_predictions",
        "dataset_sha256": content_sha256(dataset),
        "run_type": "mock",
        "metadata": {"model": "fixture-only", "measured_llm_quality": False},
        "records": [
            {
                "case_id": "supported",
                "response": {
                    "supported": True,
                    "answer": "担当者は田中です[1]。",
                    "citations": [{"id": 1, "quote": "担当者は田中です。"}],
                },
                "evidence": evidence,
            },
            {
                "case_id": "unsupported",
                "response": {
                    "supported": True,
                    "answer": "担当者は佐藤です[1]。",
                    "citations": [{"id": 1, "quote": "担当者は田中です。"}],
                },
                "evidence": evidence,
            },
            {
                "case_id": "unanswerable",
                "response": {"supported": False, "answer": "資料から納期は確認できません。", "citations": []},
                "evidence": evidence,
            },
        ],
    }
    return dataset, predictions


def test_mechanical_validity_does_not_imply_semantic_support(inputs):
    dataset, predictions = inputs
    report = evaluate_answers(dataset, predictions)
    unsupported = report["records"][1]
    assert unsupported["automatic"]["citation_validation"]["valid"]
    assert unsupported["automatic"]["semantic_support"] == "not_evaluated"
    assert unsupported["human_review_required"]
    assert report["summary"]["human"]["grounding"]["mean_score"] is None
    assert report["summary"]["human"]["grounding"]["applicable"] == 0
    assert report["summary"]["human_review_required"] == 3
    assert report["run_type"] == "mock"
    assert json.loads(json.dumps(report, ensure_ascii=False))["kind"] == "answer_grounding_report"
    assert "recall_at_k" not in report["summary"]["automatic"]


def test_missing_citation_on_additional_sentence_is_mechanical_failure(inputs):
    dataset, predictions = inputs
    predictions["records"][0]["response"]["answer"] += "納期は明日です。"
    report = evaluate_answers(dataset, predictions)
    assert not report["records"][0]["automatic"]["citation_validation"]["valid"]
    assert report["summary"]["human"]["correctness"]["mean_score"] is None


def test_false_supported_with_hallucination_is_only_a_declaration(inputs):
    dataset, predictions = inputs
    predictions["records"][2]["response"]["answer"] = "納期は明日です。"
    report = evaluate_answers(dataset, predictions)
    row = report["records"][2]
    assert row["automatic"]["declared_answerability_match"] == 1
    assert row["automatic"]["semantic_support"] == "not_evaluated"
    assert row["human_review_required"]
    assert report["summary"]["human"]["abstention"]["mean_score"] is None


def test_empty_review_template_does_not_count_as_judgment(inputs):
    dataset, predictions = inputs
    reviews = review_template(dataset, predictions)
    report = evaluate_answers(dataset, predictions, reviews)
    assert report["summary"]["human"]["complete_cases"] == 0
    assert report["summary"]["human"]["correctness"]["coverage"] == 0
    assert report["summary"]["human"]["correctness"]["mean_score"] is None
    assert reviews["dataset_sha256"] == content_sha256(dataset)


def judged_review(dataset, predictions, index=0, origin="human"):
    reviews = review_template(dataset, predictions)
    row = reviews["records"][index]
    row.update(
        {
            "origin": origin,
            "reviewer": "test reviewer",
            "correctness": "correct",
            "grounding": "supported",
            "citation_appropriateness": "appropriate",
            "abstention": "not_applicable",
            "rationale": "テスト用の原文と照合した。",
            "claims": [
                {
                    "text": "担当者は田中です。",
                    "support": "supported",
                    "evidence_ids": [1],
                    "rationale": "提示原文の担当者と一致。",
                }
            ],
        }
    )
    return reviews


def test_partial_review_coverage_and_denominators(inputs):
    dataset, predictions = inputs
    reviews = judged_review(dataset, predictions)
    reviews["records"][1].update(
        {
            "reviewer": "test reviewer",
            "correctness": "incorrect",
            "rationale": "担当者名が原文と異なる。",
        }
    )
    report = evaluate_answers(dataset, predictions, reviews)
    summary = report["summary"]["human"]
    assert summary["correctness"]["mean_score"] == 0.5
    assert summary["correctness"]["reviewed"] == 2
    assert summary["correctness"]["coverage"] == 0.6667
    assert summary["grounding"]["applicable"] == 1
    assert summary["abstention"]["applicable"] == 0
    assert summary["abstention"]["not_applicable"] == 1
    assert summary["abstention"]["mean_score"] is None
    assert summary["complete_cases"] == 1
    assert report["summary"]["human_review_required"] == 2


def test_fixture_review_is_separated_from_human_measurements(inputs):
    dataset, predictions = inputs
    reviews = judged_review(dataset, predictions, origin="fixture")
    report = evaluate_answers(dataset, predictions, reviews)
    assert report["summary"]["fixture"]["correctness"]["mean_score"] == 1
    assert report["summary"]["human"]["correctness"]["mean_score"] is None
    assert report["summary"]["human_review_required"] == 3
    predictions["run_type"] = "live"
    reviews["predictions_sha256"] = content_sha256(predictions)
    with pytest.raises(ValueError, match="fixture"):
        evaluate_answers(dataset, predictions, reviews)


@pytest.mark.parametrize("change", ["dataset", "answer", "evidence", "delivered"])
def test_stale_review_is_rejected(inputs, change):
    dataset, predictions = inputs
    reviews = judged_review(dataset, predictions)
    if change == "dataset":
        dataset["cases"][0]["question"] += "詳細も"
    elif change == "answer":
        predictions["records"][0]["response"]["answer"] = "担当者は佐藤です[1]。"
    elif change == "evidence":
        predictions["records"][0]["evidence"][0]["text"] += "追加情報。"
    else:
        predictions["records"][0]["delivered"] = {"answer": "拒否"}
    with pytest.raises(ValueError, match="変更"):
        evaluate_answers(dataset, predictions, reviews)


@pytest.mark.parametrize("bad_quote", ["担当者は佐藤です。", "", "担当者は田中です。120万円"])
def test_gold_quote_must_match_original(inputs, bad_quote):
    dataset, _ = inputs
    dataset["cases"][0]["required_claims"][0]["evidence"][0]["quote"] = bad_quote
    with pytest.raises(ValueError, match="quote"):
        validate_answer_dataset(dataset)


@pytest.mark.parametrize("change", ["empty", "duplicate", "bool_id", "answerability", "missing_claims"])
def test_invalid_gold_is_rejected(inputs, change):
    dataset, _ = inputs
    if change == "empty":
        dataset["cases"] = []
    elif change == "duplicate":
        dataset["cases"].append(copy.deepcopy(dataset["cases"][0]))
    elif change == "bool_id":
        dataset["cases"][0]["evidence"][0]["id"] = True
    elif change == "answerability":
        dataset["cases"][0]["answerable"] = "true"
    else:
        dataset["cases"][0]["required_claims"] = []
    with pytest.raises(ValueError):
        validate_answer_dataset(dataset)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "duplicate_evidence", "nan"])
def test_invalid_predictions_are_rejected(inputs, change):
    dataset, predictions = inputs
    if change == "missing":
        predictions["records"].pop()
    elif change == "duplicate":
        predictions["records"].append(copy.deepcopy(predictions["records"][0]))
    elif change == "unknown":
        predictions["records"][0]["case_id"] = "unknown"
    elif change == "duplicate_evidence":
        predictions["records"][0]["evidence"].append(copy.deepcopy(predictions["records"][0]["evidence"][0]))
    else:
        predictions["metadata"]["seconds"] = float("nan")
    with pytest.raises(ValueError):
        evaluate_answers(dataset, predictions)


@pytest.mark.parametrize("change", ["unknown", "duplicate", "bad_label", "no_reviewer", "bad_evidence"])
def test_invalid_reviews_are_rejected(inputs, change):
    dataset, predictions = inputs
    reviews = judged_review(dataset, predictions)
    row = reviews["records"][0]
    if change == "unknown":
        row["case_id"] = "unknown"
    elif change == "duplicate":
        reviews["records"].append(copy.deepcopy(row))
    elif change == "bad_label":
        row["correctness"] = "excellent"
    elif change == "no_reviewer":
        row["reviewer"] = ""
    else:
        row["claims"][0]["evidence_ids"] = [99]
    with pytest.raises(ValueError):
        evaluate_answers(dataset, predictions, reviews)


def test_malformed_model_response_is_recorded_as_failure(inputs):
    dataset, predictions = inputs
    predictions["records"][0]["response"] = {"supported": "yes", "answer": 12, "citations": "[1]"}
    report = evaluate_answers(dataset, predictions)
    auto = report["records"][0]["automatic"]
    assert auto["response_schema_valid"] is False
    assert auto["declared_answerability_match"] is None
    assert not auto["citation_validation"]["valid"]


def test_not_generated_is_not_counted_as_correct_abstention(inputs):
    dataset, predictions = inputs
    predictions["records"][2].update(
        {
            "generation_attempted": False,
            "capture_status": "no_evidence",
            "response": {},
            "evidence": [],
            "delivered": {"answer": "本文が見つかりませんでした。", "supported": False},
        }
    )
    report = evaluate_answers(dataset, predictions)
    auto = report["records"][2]["automatic"]
    assert auto["generation_attempted"] is False
    assert auto["declared_answerability_match"] is None
    assert auto["response_schema_valid"] is None
    assert auto["citation_validation"] is None
    assert report["summary"]["automatic"]["declared_answerability_denominator"] == 2
    assert report["records"][2]["human_review_required"]


def test_evaluation_does_not_mutate_inputs(inputs):
    dataset, predictions = inputs
    original = copy.deepcopy(inputs)
    evaluate_answers(dataset, predictions)
    assert inputs == original


def test_prediction_is_bound_to_exact_dataset(inputs):
    dataset, predictions = inputs
    dataset["cases"][0]["question"] = "費用はいくらですか？"
    with pytest.raises(ValueError, match="データセットが変更"):
        evaluate_answers(dataset, predictions)


def test_prediction_dataset_hash_is_required(inputs):
    dataset, predictions = inputs
    del predictions["dataset_sha256"]
    with pytest.raises(ValueError, match="データセットが変更"):
        evaluate_answers(dataset, predictions)


@pytest.mark.parametrize("change", ["run_type", "metadata"])
def test_review_is_bound_to_run_provenance(inputs, change):
    dataset, predictions = inputs
    reviews = judged_review(dataset, predictions)
    if change == "run_type":
        predictions[change] = "live"
    else:
        predictions["metadata"]["model"] = "different-model"
    with pytest.raises(ValueError, match="実行条件が変更"):
        evaluate_answers(dataset, predictions, reviews)


@pytest.mark.parametrize("field", ["run_type", "origin", "support"])
def test_non_scalar_enum_reports_value_error(inputs, field):
    dataset, predictions = inputs
    reviews = judged_review(dataset, predictions)
    if field == "run_type":
        predictions[field] = []
    elif field == "origin":
        reviews["records"][0][field] = []
    else:
        reviews["records"][0]["claims"][0][field] = []
    with pytest.raises(ValueError):
        evaluate_answers(dataset, predictions, reviews)


@pytest.mark.parametrize("source_hash", ["abcd", 1, [], "g" * 64])
def test_invalid_source_hash_is_rejected(inputs, source_hash):
    dataset, _ = inputs
    dataset["cases"][0]["source_sha256"] = source_hash
    with pytest.raises(ValueError, match="source_sha256"):
        validate_answer_dataset(dataset)


def test_hash_ignores_key_order_but_not_content():
    assert content_sha256({"b": 2, "a": 1}) == content_sha256({"a": 1, "b": 2})
    assert content_sha256({"b": 3, "a": 1}) != content_sha256({"a": 1, "b": 2})
