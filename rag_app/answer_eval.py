"""Offline answer scoring and optional capture through the real answer() path.

Unlike `rag_app evaluate`, score/review-template need no search server or model.
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_app.answer_evaluation import (
    content_sha256,
    evaluate_answers,
    review_template,
    validate_answer_dataset,
)


def capture_answers(index, dataset, ollama, config, *, run_type):
    """Capture raw generation *and* the answer actually delivered to the user.

    This runs against a prebuilt index; it never builds/replaces an index or changes
    model settings. The caller must supply a real model client for a live run.
    """
    from rag_app.documents import fingerprint
    from rag_app.service import answer

    validate_answer_dataset(dataset)
    if run_type not in ("mock", "live", "captured"):
        raise ValueError("run_type は mock/live/captured です。")
    files = {row["relpath"]: row for row in index.files()}
    missing = {case["source"] for case in dataset["cases"]} - files.keys()
    if missing:
        raise ValueError("索引にない評価資料: " + ", ".join(sorted(missing)))
    source_hashes = {}
    for case in dataset["cases"]:
        source = case["source"]
        if source not in source_hashes:
            source_hashes[source] = fingerprint(Path(files[source]["path"]))
        if case.get("source_sha256") and case["source_sha256"] != source_hashes[source]:
            raise ValueError(f"正解作成時から評価資料が変更されています: {source}")
    records = []
    for case in dataset["cases"]:
        started = time.perf_counter()
        try:
            result = answer(index, files[case["source"]]["id"], case["question"], ollama, config)
            raw = result.get("raw_response")
            record = {
                "case_id": case["id"],
                "response": raw if isinstance(raw, dict) else {},
                "evidence": result.get("retrieved", []),
                "generation_attempted": raw is not None,
                "capture_status": result["status"],
                "delivered": result,
            }
        except (ValueError, OSError, RuntimeError) as exc:
            # A failed request is not a successful refusal or a quality observation.
            record = {
                "case_id": case["id"],
                "response": {},
                "evidence": [],
                "generation_attempted": False,
                "capture_status": "error",
                "error": str(exc),
            }
        record["seconds"] = round(time.perf_counter() - started, 6)
        records.append(record)
    return {
        "schema_version": 1,
        "kind": "answer_predictions",
        "dataset_sha256": content_sha256(dataset),
        "run_type": run_type,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "rag_app.service.answer",
            "scope": "single_document_retrieval_and_generation",
            "python": platform.python_version(),
            "config": dict(config),
            "source_sha256": source_hashes,
            "quality_measured": False,
            "note": "生成を記録しただけです。意味的な品質は独立した人手レビューが必要です。",
        },
        "records": records,
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="検索指標から独立した回答・根拠評価")
    parser.add_argument("action", choices=["score", "review-template", "run"])
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action != "run" and not args.predictions:
        parser.error("score/review-template には --predictions が必要です。")
    if args.action == "run" and not args.folder:
        parser.error("run には事前に索引を作成した --folder が必要です。")
    if args.reviews and args.action != "score":
        parser.error("--reviews は score にのみ指定できます。")
    if args.action == "run" and args.predictions:
        parser.error("run は新しい予測を --output に保存します。--predictions は指定しません。")
    inputs = [args.dataset, args.predictions, args.reviews]
    if args.output.resolve() in {path.resolve() for path in inputs if path}:
        parser.error("--output は入力ファイルと別のパスにしてください。")
    try:
        dataset = read_json(args.dataset)
        if args.action == "run":
            # Lazy imports keep offline scoring independent of FAISS/Meilisearch.
            from rag_app.meili_faiss import MeiliFaissIndex, database_path, settings
            from rag_app.models import Ollama

            folder = args.folder.expanduser().resolve()
            path = database_path(folder)
            if not path.is_file():
                raise ValueError("評価用索引がありません。先に rag_app index を実行してください。")
            config = settings() | {"folder": str(folder)}
            with MeiliFaissIndex(path) as index:
                result = capture_answers(index, dataset, Ollama(), config, run_type="live")
        else:
            predictions = read_json(args.predictions)
            if args.action == "review-template":
                result = review_template(dataset, predictions)
            else:
                reviews = read_json(args.reviews) if args.reviews else None
                result = evaluate_answers(dataset, predictions, reviews)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"エラー: {exc}\n")


if __name__ == "__main__":
    main()
