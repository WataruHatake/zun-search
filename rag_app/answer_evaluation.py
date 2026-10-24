"""回答評価。構文検査と、人が正解・原文を読んで行う意味判定を分離する。

このモジュールはモデルや検索エンジンを起動しない。``supported`` はモデルの
自己申告であり、正確性・根拠整合性・適切な回答拒否の証明には使わない。
"""

import hashlib
import json
import re
import statistics
from datetime import datetime, timezone

from rag_app.citations import validate_citations

AXES = {
    "correctness": {"correct": 1.0, "partial": 0.5, "incorrect": 0.0},
    "grounding": {"supported": 1.0, "partial": 0.5, "unsupported": 0.0},
    "citation_appropriateness": {"appropriate": 1.0, "partial": 0.5, "inappropriate": 0.0},
    "abstention": {"appropriate": 1.0, "inappropriate": 0.0},
}


def content_sha256(value):
    """JSONのキー順によらない内容ハッシュ。NaN等の非JSON値は受け付けない。"""
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("評価入力は標準JSONとして保存できる値にしてください。") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _envelope(data, kind):
    _require(isinstance(data, dict), f"{kind}: JSONオブジェクトが必要です。")
    _require(
        type(data.get("schema_version")) is int and data["schema_version"] == 1, "schema_version は1です。"
    )
    _require(data.get("kind") == kind, f"kind は {kind} です。")
    content_sha256(data)


def _evidence(rows, label):
    _require(isinstance(rows, list), f"{label}: evidence は配列です。")
    seen = set()
    for row in rows:
        _require(isinstance(row, dict), f"{label}: evidence の各項目はオブジェクトです。")
        identifier = row.get("id")
        _require(type(identifier) is int and identifier > 0, f"{label}: evidence.id は正の整数です。")
        _require(identifier not in seen, f"{label}: evidence.id が重複しています。")
        seen.add(identifier)
        _require(_text(row.get("text")), f"{label}: evidence.text が必要です。")
        _require(_text(row.get("location")), f"{label}: evidence.location が必要です。")
    return {row["id"]: row for row in rows}


def validate_answer_dataset(dataset):
    """検索評価データとは別の、出典付き正解データを検証して返す。"""
    _envelope(dataset, "answer_grounding")
    provenance = dataset.get("provenance")
    _require(
        isinstance(provenance, dict)
        and _text(provenance.get("type"))
        and _text(provenance.get("description")),
        "provenance に type と description が必要です。",
    )
    cases = dataset.get("cases")
    _require(isinstance(cases, list) and 1 <= len(cases) <= 200, "cases は1〜200問の配列です。")
    seen = set()
    for case in cases:
        _require(isinstance(case, dict), "各ケースはオブジェクトです。")
        case_id = case.get("id")
        _require(_text(case_id), "ケースの id が必要です。")
        _require(case_id not in seen, f"ケースIDが重複しています: {case_id}")
        seen.add(case_id)
        for key in ("question", "source", "reference_answer"):
            _require(_text(case.get(key)), f"{case_id}: {key} が必要です。")
        if "source_sha256" in case:
            _require(
                isinstance(case["source_sha256"], str)
                and re.fullmatch(r"[0-9a-fA-F]{64}", case["source_sha256"]) is not None,
                f"{case_id}: source_sha256 は64桁の16進数です。",
            )
        _require(type(case.get("answerable")) is bool, f"{case_id}: answerable は真偽値です。")
        evidence = _evidence(case.get("evidence"), case_id)
        claims = case.get("required_claims")
        _require(isinstance(claims, list), f"{case_id}: required_claims は配列です。")
        _require(bool(claims) == case["answerable"], f"{case_id}: 回答可能なケースだけに正解主張が必要です。")
        claim_ids = set()
        for claim in claims:
            _require(isinstance(claim, dict), f"{case_id}: 正解主張はオブジェクトです。")
            identifier = claim.get("id")
            _require(_text(identifier), f"{case_id}: 正解主張の id が必要です。")
            _require(identifier not in claim_ids, f"{case_id}: 正解主張IDが重複しています。")
            claim_ids.add(identifier)
            _require(_text(claim.get("text")), f"{case_id}: 正解主張の text が必要です。")
            references = claim.get("evidence")
            _require(isinstance(references, list) and references, f"{case_id}: 正解主張の根拠が必要です。")
            for reference in references:
                _require(isinstance(reference, dict), f"{case_id}: 正解根拠はオブジェクトです。")
                source_id = reference.get("source_id")
                _require(
                    type(source_id) is int and source_id in evidence, f"{case_id}: 正解根拠IDが不正です。"
                )
                quote = reference.get("quote")
                _require(
                    _text(quote) and quote in evidence[source_id]["text"],
                    f"{case_id}: 正解の quote は原文の連続した部分と完全一致する必要があります。",
                )
        forbidden = case.get("forbidden_claims")
        _require(
            isinstance(forbidden, list) and all(_text(x) for x in forbidden),
            f"{case_id}: forbidden_claims が不正です。",
        )
    return dataset


def _predictions(dataset, predictions):
    validate_answer_dataset(dataset)
    _envelope(predictions, "answer_predictions")
    _require(
        predictions.get("dataset_sha256") == content_sha256(dataset),
        "予測のデータセットが変更されています。対象データセットで予測を作り直してください。",
    )
    _require(
        isinstance(predictions.get("run_type"), str)
        and predictions["run_type"] in {"mock", "live", "captured"},
        "run_type は mock/live/captured です。",
    )
    _require(isinstance(predictions.get("metadata"), dict), "予測の metadata が必要です。")
    records = predictions.get("records")
    _require(isinstance(records, list), "予測の records は配列です。")
    known = {case["id"] for case in dataset["cases"]}
    indexed = {}
    for record in records:
        _require(isinstance(record, dict), "予測の各項目はオブジェクトです。")
        case_id = record.get("case_id")
        _require(_text(case_id) and case_id in known, "予測に未知の case_id があります。")
        _require(case_id not in indexed, f"予測の case_id が重複しています: {case_id}")
        _require(isinstance(record.get("response"), dict), f"{case_id}: response はオブジェクトです。")
        _evidence(record.get("evidence"), case_id)
        if "generation_attempted" in record:
            _require(type(record["generation_attempted"]) is bool, "generation_attempted は真偽値です。")
            _require(
                record["generation_attempted"] or not record["response"],
                "未生成の response は空にしてください。",
            )
        if "delivered" in record:
            _require(isinstance(record["delivered"], dict), "delivered はオブジェクトです。")
        indexed[case_id] = record
    _require(
        set(indexed) == known,
        "全ケースの予測が必要です。未実行ケースも空の response と実行状態で記録してください。",
    )
    return indexed


def review_template(dataset, predictions):
    """空欄のレビュー票。採点者名と根拠を記入するまで意味評価は未実施。"""
    indexed = _predictions(dataset, predictions)
    return {
        "schema_version": 1,
        "kind": "answer_reviews",
        "dataset_sha256": content_sha256(dataset),
        "predictions_sha256": content_sha256(predictions),
        "instructions": (
            "評価対象はresponseの生の生成回答。deliveredはアプリの拒否・整形後の参考情報です。"
            "原文・正解主張・禁止主張と照合し、全主張をclaimsへ分解してください。"
            "引用の一致だけで正確性を判定せず、数値・否定・条件・時点・推論も確認します。"
            "supported=falseでも本文が回答を捏造していないか確認してください。"
            "nullは未確認、not_applicableは確認の結果その軸が適用外である場合だけ使います。"
            "回答不能ではabstention、回答可能でも不当に拒否していないかabstentionを採点します。"
            "reviewerとrationaleを記入してください。fixtureはモックの期待判定で、実測ではありません。"
        ),
        "records": [
            {
                "case_id": case["id"],
                "prediction_sha256": content_sha256(indexed[case["id"]]),
                "reviewer": "",
                "origin": "human",
                **{axis: None for axis in AXES},
                "rationale": "",
                "claims": [],
            }
            for case in dataset["cases"]
        ],
    }


def _reviews(dataset, predictions, indexed, reviews):
    if reviews is None:
        return {}
    _envelope(reviews, "answer_reviews")
    _require(
        reviews.get("dataset_sha256") == content_sha256(dataset), "レビューのデータセットが変更されています。"
    )
    _require(
        reviews.get("predictions_sha256") == content_sha256(predictions),
        "レビューの予測または実行条件が変更されています。",
    )
    rows = reviews.get("records")
    _require(isinstance(rows, list), "レビューの records は配列です。")
    result = {}
    for row in rows:
        _require(isinstance(row, dict), "レビューの各項目はオブジェクトです。")
        case_id = row.get("case_id")
        _require(_text(case_id) and case_id in indexed, "レビューに未知の case_id があります。")
        _require(case_id not in result, f"レビューの case_id が重複しています: {case_id}")
        _require(
            row.get("prediction_sha256") == content_sha256(indexed[case_id]),
            f"{case_id}: 回答または提示原文が変更されています。",
        )
        origin = row.get("origin")
        _require(
            isinstance(origin, str) and origin in {"human", "fixture"},
            "レビューの origin は human/fixture です。",
        )
        _require(
            origin != "fixture" or predictions["run_type"] == "mock",
            "fixtureレビューはmock実行にしか適用できません。",
        )
        for axis, labels in AXES.items():
            value = row.get(axis)
            _require(
                value is None or isinstance(value, str) and value in {*labels, "not_applicable"},
                f"{case_id}: {axis} の判定値が不正です。",
            )
        claims = row.get("claims")
        _require(isinstance(claims, list), f"{case_id}: claims は配列です。")
        has_judgment = bool(claims) or any(row.get(axis) is not None for axis in AXES)
        if has_judgment:
            _require(
                _text(row.get("reviewer")) and _text(row.get("rationale")),
                f"{case_id}: 採点者と判定理由が必要です。",
            )
        else:
            _require(
                isinstance(row.get("reviewer"), str) and isinstance(row.get("rationale"), str),
                "レビューの記入欄が不正です。",
            )
        evidence_ids = {item["id"] for item in indexed[case_id]["evidence"]}
        for claim in claims:
            _require(isinstance(claim, dict), "レビュー主張はオブジェクトです。")
            _require(
                _text(claim.get("text")) and _text(claim.get("rationale")),
                "レビュー主張には本文と判定理由が必要です。",
            )
            _require(
                isinstance(claim.get("support"), str)
                and claim["support"] in {"supported", "unsupported", "contradicted"},
                "レビュー主張の support が不正です。",
            )
            ids = claim.get("evidence_ids")
            _require(
                isinstance(ids, list) and all(type(i) is int and i in evidence_ids for i in ids),
                "レビュー主張の evidence_ids が不正です。",
            )
            _require(len(ids) == len(set(ids)), "レビュー主張の evidence_ids が重複しています。")
            _require(claim["support"] != "supported" or bool(ids), "supportedな主張には根拠IDが必要です。")
        result[case_id] = row
    return result


def _mean(values):
    return round(statistics.mean(values), 4) if values else None


def _semantic_summary(reviews, origin, total):
    selected = [r for r in reviews.values() if r["origin"] == origin]
    summary = {
        "reviewed_cases": sum(any(r.get(axis) is not None for axis in AXES) for r in selected),
        "complete_cases": sum(all(r.get(axis) is not None for axis in AXES) for r in selected),
        "total_cases": total,
    }
    for axis, labels in AXES.items():
        reviewed = [r[axis] for r in selected if r.get(axis) is not None]
        values = [labels[value] for value in reviewed if value != "not_applicable"]
        summary[axis] = {
            "mean_score": _mean(values),
            "fully_satisfied_rate": _mean([float(value == 1) for value in values]),
            "reviewed": len(reviewed),
            "applicable": len(values),
            "not_applicable": len(reviewed) - len(values),
            "coverage": round(len(reviewed) / total, 4),
        }
    return summary


def evaluate_answers(dataset, predictions, reviews=None):
    """機械検査とレビュー結果を、検索指標を混ぜずJSON保存可能な形で返す。"""
    indexed = _predictions(dataset, predictions)
    reviewed = _reviews(dataset, predictions, indexed, reviews)
    records = []
    for case in dataset["cases"]:
        prediction = indexed[case["id"]]
        response = prediction["response"]
        attempted = prediction.get("generation_attempted", True)
        supported = response.get("supported")
        schema_valid = (
            (
                type(supported) is bool
                and isinstance(response.get("answer"), str)
                and isinstance(response.get("citations"), list)
            )
            if attempted
            else None
        )
        # Abstention is a self-report only. An answer can be hidden behind supported=false.
        declared_match = (
            float(supported == case["answerable"]) if attempted and type(supported) is bool else None
        )
        if not attempted or (
            supported is False and not response.get("answer") and not response.get("citations")
        ):
            citation_validation = None
        else:
            citation_validation = validate_citations(
                response.get("answer"), response.get("citations"), prediction["evidence"]
            )
        review = reviewed.get(case["id"])
        completed_human = bool(
            review and review["origin"] == "human" and all(review.get(axis) is not None for axis in AXES)
        )
        records.append(
            {
                "case_id": case["id"],
                "question": case["question"],
                "source": case["source"],
                "gold": {
                    key: case[key]
                    for key in (
                        "answerable",
                        "reference_answer",
                        "required_claims",
                        "forbidden_claims",
                        "evidence",
                    )
                },
                "prediction_sha256": content_sha256(prediction),
                "prediction": prediction,
                "automatic": {
                    "generation_attempted": attempted,
                    "response_schema_valid": schema_valid,
                    "declared_answerability_match": declared_match,
                    "citation_validation": citation_validation,
                    "semantic_support": "not_evaluated",
                },
                "review": review,
                "human_review_required": not completed_human,
            }
        )
    automatic = [row["automatic"] for row in records]
    citation_results = [
        row["automatic"]["citation_validation"]
        for row in records
        if row["prediction"]["response"].get("supported") is True
        and row["automatic"]["citation_validation"] is not None
    ]
    declarations = [
        row["declared_answerability_match"]
        for row in automatic
        if row["declared_answerability_match"] is not None
    ]
    schemas = [
        float(row["response_schema_valid"]) for row in automatic if row["response_schema_valid"] is not None
    ]
    return {
        "schema_version": 1,
        "kind": "answer_grounding_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": content_sha256(dataset),
        "predictions_sha256": content_sha256(predictions),
        "run_type": predictions["run_type"],
        "metadata": predictions["metadata"],
        "dataset_provenance": dataset["provenance"],
        "scope": "raw_generation",
        "limitations": [
            "引用ID・引用原文・配置の構文検査は意味的な裏付けを保証しない。",
            "declared_answerability_match はsupported自己申告と正解ラベルの一致であり、正しい回答・拒否の判定ではない。",
            "意味評価は人手で原文、必要な主張、禁止主張、回答不能時の振る舞いを確認する。",
            "未確認の意味指標はnull。fixtureの期待判定はhumanと分離し、実LLM品質の実測に使わない。",
            "評価対象は生のresponse。アプリが表示したdeliveredの拒否結果とは別である。",
        ],
        "metric_definitions": {
            "semantic_mean_score": "人手ラベル: 完全=1、一部=0.5、不適切=0。not_applicable/nullを母数から除外。",
            "semantic_coverage": "各軸の記入済みケース数（not_applicableを含む）/全ケース数。",
            "citation_checks": "supported=trueの応答だけを母数にする。拒否文章の構文検査は個別レコードに残し、拒否の適切さは人手のabstentionで別判定。",
        },
        "summary": {
            "cases": len(records),
            "automatic": {
                "response_schema_valid_rate": _mean(schemas),
                "response_schema_denominator": len(schemas),
                "citation_checks_pass_rate": _mean([float(row["valid"]) for row in citation_results]),
                "citation_checks_denominator": len(citation_results),
                "citation_checks_scope": "declared_supported_responses",
                "declared_answerability_match_rate": _mean(declarations),
                "declared_answerability_denominator": len(declarations),
            },
            "human": _semantic_summary(reviewed, "human", len(records)),
            "fixture": _semantic_summary(reviewed, "fixture", len(records)),
            "human_review_required": sum(row["human_review_required"] for row in records),
        },
        "records": records,
    }
