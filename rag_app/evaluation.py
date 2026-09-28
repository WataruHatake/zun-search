"""正解ファイルに対する候補回収率と順位を別々に評価する。"""

import math
import statistics

from rag_app.service import search


def validate_dataset(data):
    if not isinstance(data, list) or not data:
        raise ValueError("評価データは空でないJSON配列にしてください。")
    if len(data) > 200:
        raise ValueError("1回の評価は200問までです。")
    for row in data:
        if not isinstance(row, dict) or not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError("各問題に query が必要です。")
        if not isinstance(row.get("relevant_files"), list) or not all(
            isinstance(p, str) for p in row["relevant_files"]
        ):
            raise ValueError("relevant_files に正解資料の相対パスの配列が必要です。該当なしは [] です。")
    return data


def metrics(ranked, candidates, relevant, k=5):
    relevant = set(relevant)
    if not relevant:
        return {
            "candidate_recall": None,
            "recall_at_k": None,
            "mrr": None,
            "ndcg_at_k": None,
            "no_match_correct": float(not ranked),
        }
    ranks = [i for i, path in enumerate(ranked, 1) if path in relevant]
    dcg = sum(1 / math.log2(i + 1) for i in ranks if i <= k)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return {
        "candidate_recall": len(set(candidates) & relevant) / len(relevant),
        "recall_at_k": len(set(ranked[:k]) & relevant) / len(relevant),
        "mrr": 1 / min(ranks) if ranks else 0.0,
        "ndcg_at_k": dcg / ideal,
        "no_match_correct": None,
    }


def evaluate(index, dataset, ollama, config, modes, k=5, progress=None):
    validate_dataset(dataset)
    known_files = {f["relpath"] for f in index.files()}
    unknown = {p for row in dataset for p in row["relevant_files"]} - known_files
    if unknown:
        raise ValueError("索引にない正解ファイルがあります: " + ", ".join(sorted(unknown)[:10]))
    records = []
    for mode in modes:
        for row in dataset:
            if progress:
                progress(len(records), len(modes) * len(dataset), f"{mode}: {row['query']}")
            result = search(index, row["query"], ollama, config, mode)
            ranked = [f["relpath"] for f in result["files"]]
            records.append(
                {
                    "mode": mode,
                    "query": row["query"],
                    "split": row.get("split", "unspecified"),
                    **metrics(ranked, result["candidate_files"], row["relevant_files"], k),
                    "ranked_files": ranked,
                    "candidate_files": result["candidate_files"],
                    "candidate_ids": result["candidate_ids"],
                    "reranked_ids": result["reranked_ids"],
                    "seconds": result["seconds"],
                    "warnings": result["warnings"],
                    "vector_used": result["vector_used"],
                    "reranker_used": result["reranker_used"],
                    "expansions": result["expansions"],
                    "intent": result["intent"],
                    "formats": result["formats"],
                    "retrieval": result["retrieval"],
                }
            )
    summary = []
    for mode in modes:
        for split in dict.fromkeys(r["split"] for r in records if r["mode"] == mode):
            selected = [r for r in records if r["mode"] == mode and r["split"] == split]
            averages = {}
            for key in ("candidate_recall", "recall_at_k", "mrr", "ndcg_at_k", "no_match_correct", "seconds"):
                values = [r[key] for r in selected if r[key] is not None]
                averages[key] = round(statistics.mean(values), 4) if values else None
            summary.append(
                {
                    "mode": mode,
                    "split": split,
                    "queries": len(selected),
                    **averages,
                    "runs_with_warnings": sum(bool(r["warnings"]) for r in selected),
                }
            )
    return {"k": k, "config": config, "summary": summary, "records": records}
