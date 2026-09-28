import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from rag_app.config import DEFAULTS
from rag_app.evaluation import evaluate, metrics
from rag_app.index import Index, terms
from rag_app.models import ModelError, Ollama
from rag_app.service import answer, expand_query, llm_rerank, original_quote, retrieve, search, unique_queries


class FakeOllama:
    def __init__(self):
        self.signature = "embedding:v1"
        self.calls = 0
        self.fail_after = None
        self.reply = None
        self.last_payload = None
        self.chat_keep_alives = []

    def verify_local(self, model):
        return self.signature

    def request(self, route, payload):
        assert route == "/api/show"
        return {"capabilities": ["vision"]}

    def embed(self, model, texts, query=False):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ModelError("テスト用の接続中断")
        vectors = np.array(
            [[1.0 if "RAG" in text else 0.0, 1.0 if "休暇" in text else 0.0, 0.1] for text in texts],
            dtype=np.float32,
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def chat_json(self, model, system, prompt, schema, keep_alive=0):
        self.chat_keep_alives.append(keep_alive)
        self.last_payload = json.loads(prompt)
        if self.reply is not None and ("queries" not in self.reply or "original_query" in self.last_payload):
            if "original_query" in self.last_payload:
                return {"intent": self.last_payload["original_query"], "formats": [], **self.reply}
            return self.reply
        if "candidates" in self.last_payload:
            return {
                "scores": [
                    {"id": c["id"], "reason": "本文で検索対象を確認", "score": 3}
                    for c in self.last_payload["candidates"]
                ]
            }
        evidence = self.last_payload["evidence"][0]
        return {
            "supported": True,
            "answer": "資料の内容です。[1]",
            "citations": [{"id": 1, "quote": evidence["text"][:30]}],
        }


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "青空商事.txt").write_text("青空商事でRAGを実装した。担当は田中。費用は120万円。")
    (folder / "休暇.md").write_text("有給休暇は前営業日までに申請する。")
    with Index(tmp_path / "test.sqlite3") as index:
        model = FakeOllama()
        index.update(folder, model, "embed")
        yield folder, index, model


def test_japanese_filename_text_and_empty_query(corpus):
    _, index, model = corpus
    result = search(index, "青空商事 RAG 実装", model, DEFAULTS, "keyword")
    assert result["files"][0]["relpath"] == "青空商事.txt"
    assert len({f["file_id"] for f in result["files"]}) == len(result["files"])
    assert not search(index, "火星燃料タンク", model, DEFAULTS, "keyword")["files"]
    with pytest.raises(ValueError):
        search(index, "  ", model, DEFAULTS)


def test_generic_verbs_do_not_match_unrelated_documents(corpus):
    folder, index, model = corpus
    assert "為る" not in terms("RAGを実装した報告を探してください")
    (folder / "git.txt").write_text("Gitの基本操作を確認する。")
    with index.db:
        index.set_meta("tokenizer", "1")
    report = index.update(folder, model, "embed")
    assert report["retokenized"] == 2 and report["embedded"] == 1
    result = search(index, "RAGを実装した報告", model, DEFAULTS, "keyword")
    assert "git.txt" not in [f["relpath"] for f in result["files"]]


def test_changes_same_size_timestamp_move_and_delete(corpus):
    folder, index, model = corpus
    path = folder / "青空商事.txt"
    before = path.stat()
    path.write_text(path.read_text().replace("田中", "鈴木"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    report = index.update(folder, model, "embed")
    assert report["changed"] == 1
    assert not index.keyword("田中")
    assert index.keyword("鈴木")
    path.rename(folder / "新しい報告.txt")
    (folder / "休暇.md").unlink()
    report = index.update(folder, model, "embed")
    assert report["removed"] == 2 and report["added"] == 1
    assert [f["relpath"] for f in index.files()] == ["新しい報告.txt"]
    assert not index.keyword("休暇")


def test_missing_root_preserves_index(corpus):
    folder, index, model = corpus
    previous = index.stats()
    with pytest.raises(ValueError):
        index.update(folder / "missing", model, "embed")
    assert index.stats() == previous


def test_corrupt_changed_document_has_no_stale_text(tmp_path):
    from reportlab.pdfgen import canvas

    folder = tmp_path / "docs"
    folder.mkdir()
    path = folder / "sample.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawString(40, 700, "unique_previous_content")
    pdf.save()
    with Index(tmp_path / "db.sqlite3") as index:
        index.update(folder)
        assert index.keyword("previous")
        path.write_bytes(b"not a PDF")
        report = index.update(folder)
        assert report["errors"] == 1
        assert not index.keyword("previous")
        assert index.files()[0]["status"] == "error"


def test_interrupted_embedding_resumes_and_changed_digest_invalidates(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for number in range(10):
        (folder / f"{number}.txt").write_text(f"RAGの資料 {number}")
    with Index(tmp_path / "index.sqlite3") as index:
        model = FakeOllama()
        model.fail_after = 1
        report = index.update(folder, model, "embed")
        assert report["embedded"] == 8 and report["warnings"]
        model.fail_after = None
        assert index.update(folder, model, "embed")["embedded"] == 2
        assert index.update(folder, model, "embed")["embedded"] == 0
        model.signature = "embedding:v2"
        assert index.update(folder, model, "embed")["embedded"] == 10


def test_long_file_does_not_consume_all_candidates(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "long.txt").write_text("RAG検索の実装。\n" * 1000)
    (folder / "short.txt").write_text("RAG検索を実装した結果。")
    with Index(tmp_path / "db.sqlite3") as index:
        model = FakeOllama()
        index.update(folder, model, "embed")
        hits = index.keyword("RAG", 4)
        assert len({r["file_id"] for r in index.chunks(ids=[h[0] for h in hits])}) == 2
        vectors = index.vector(model.embed("embed", ["RAG"])[0], 4)
        assert len({r["file_id"] for r in index.chunks(ids=[h[0] for h in vectors])}) == 2

        config = {**DEFAULTS, "rerank_limit": 2}
        baseline = search(index, "RAG", model, config)
        for mode in ("dedicated", "llm", "expanded"):
            with (
                patch(
                    "rag_app.service.expand_query",
                    return_value={"intent": "RAG", "formats": [], "queries": []},
                ),
                patch("rag_app.service.dedicated_rerank", return_value=[0.8, 0.8]),
            ):
                result = search(index, "RAG", model, config, mode)
            assert not result["warnings"]
            assert result["candidate_ids"] == baseline["candidate_ids"]
            reranked = index.chunks(ids=result["reranked_ids"])
            assert len(reranked) == len({r["file_id"] for r in reranked}) == 2
            assert {p["id"] for f in result["files"] for p in f["passages"]} == set(result["candidate_ids"])
            if mode in {"llm", "expanded"}:
                longest_passage = max(len(p["text"]) for f in result["files"] for p in f["passages"])
                assert any(len(c["text"]) > longest_passage for c in model.last_payload["candidates"])


def test_baseline_and_rerankers_use_same_candidates(corpus, monkeypatch):
    _, index, model = corpus
    monkeypatch.setattr("rag_app.service.dedicated_rerank", lambda query, texts, name: [0.8] * len(texts))
    results = [search(index, "RAG", model, DEFAULTS, mode) for mode in ["hybrid", "dedicated", "llm"]]
    assert results[0]["candidate_ids"] == results[1]["candidate_ids"] == results[2]["candidate_ids"]
    assert all(not result["warnings"] for result in results)


def test_expansion_retains_original_query(corpus):
    _, index, model = corpus
    model.reply = {"queries": ["検索拡張生成", "検索拡張生成"]}
    result = search(index, "RAG", model, DEFAULTS, "expanded")
    assert result["query"] == "RAG"
    assert result["expansions"] == ["検索拡張生成"]
    assert "全文" in result["files"][0]["matches"]

    # 元の文の表記違いで、検索回数や同じ根拠への加点が増えない。
    baseline = search(index, "RAG", model, {**DEFAULTS, "min_similarity": 0.0}, "hybrid")
    model.reply = {"queries": ["rag", "ＲＡＧ"]}
    with patch.object(index, "keyword", wraps=index.keyword) as keyword:
        result = search(index, "RAG", model, DEFAULTS, "expanded")
        assert keyword.call_count == 1
    assert not result["expansions"]
    assert [f["score"] for f in result["files"]] == [f["score"] for f in baseline["files"]]
    model.reply = {"queries": ["RAG  事例", "ｒａｇ　事例"]}
    assert expand_query("RAG", model, "test")["queries"] == ["RAG  事例"]


def test_expansion_does_not_invent_implementation_stage():
    model = FakeOllama()
    model.reply = {"queries": ["燃料タンクの設計", "燃料タンクの実装済み報告"]}
    assert expand_query("燃料タンクの設計", model, "test")["queries"] == []


def test_query_deduplication_and_empty_vector_results(corpus):
    folder, index, model = corpus
    (folder / "local.txt").write_text("ローカルAIのモデルを動かす資料。")
    index.update(folder, model, "embed")
    assert unique_queries(["ローカルAI", "ローカル AI", "ローカルａｉ"]) == ["ローカルAI"]
    assert len(unique_queries(["nowhere", "now here", "C++", "C#"])) == 4
    baseline, _, _ = retrieve(index, "ローカルAI", model, "embed", use_vectors=False)
    with patch.object(index, "keyword", wraps=index.keyword) as keyword:
        rows, _, info = retrieve(
            index,
            "ローカルAI",
            model,
            "embed",
            use_vectors=False,
            expansions=["ローカル AI", "ローカルAI", "ローカル AI AI"],
        )
    assert keyword.call_count == info["keyword_queries"] == 1
    assert [(r["id"], r["score"]) for r in rows] == [(r["id"], r["score"]) for r in baseline]
    with patch.object(index, "vector", return_value=[]):
        result = search(index, "ローカルAI", model, DEFAULTS, "hybrid")
    assert not result["vector_used"] and result["retrieval"]["vector_chunks"] == 0
    assert any("ベクトル検索の候補は0件" in w for w in result["warnings"])


def test_ai_search_excludes_rejected_and_unchecked_files(corpus):
    _, index, model = corpus
    config = {**DEFAULTS, "rerank_limit": 1, "min_similarity": 0.95}
    with (
        patch("rag_app.service.expand_query", return_value={"intent": "RAG", "formats": [], "queries": []}),
        patch(
            "rag_app.service.llm_rerank", return_value=[{"rerank_score": 0, "rerank_reason": "対象が違う"}]
        ),
    ):
        result = search(index, "RAG", model, config, "expanded")
    assert result["retrieval"]["min_similarity"] == 0
    assert len(result["candidate_files"]) == 2 and len(result["reranked_ids"]) == 1
    assert not result["files"]  # 却下した資料も、未確認の残りの資料も紛れ込まない。
    with (
        patch("rag_app.service.expand_query", return_value={"intent": "RAG", "formats": [], "queries": []}),
        patch("rag_app.service.llm_rerank", side_effect=ModelError("応答不正")),
    ):
        with pytest.raises(ModelError, match="本文の関連性"):
            search(index, "RAG", model, config, "expanded")


def test_llm_judgments_require_valid_ids_scores_and_reasons(corpus):
    _, index, model = corpus
    rows = index.chunks()
    llm_rerank("RAG", rows + [{**rows[0], "id": -1}], model, "test")
    assert model.chat_keep_alives == ["2m", 0]  # 途中で再読込せず、最後に解放する。
    model.reply = {
        "scores": [{"id": row["id"], "reason": "本文の根拠", "score": i} for i, row in enumerate(rows)]
    }
    model.reply["scores"].reverse()
    assert [s["rerank_score"] for s in llm_rerank("RAG", rows, model, "test")] == [0, 1 / 3]
    good = model.reply["scores"][0].copy()
    for invalid in ({"id": []}, {"id": -1}, {"score": True}, {"score": 1.5}, {"score": 4}, {"reason": ""}):
        model.reply["scores"][0] = good | invalid
        with pytest.raises(ModelError):
            llm_rerank("RAG", rows, model, "test")


def test_ai_formats_are_applied_before_review_without_keyword_rules(corpus):
    _, index, model = corpus
    for extension, expected in [(".md", "休暇.md"), (".txt", "青空商事.txt")]:
        model.reply = {"queries": [], "formats": [extension], "intent": "指定形式の資料"}
        result = search(index, "RAG", model, {**DEFAULTS, "rerank_limit": 1}, "expanded")
        assert result["candidate_files"] == [expected]
        assert [f["relpath"] for f in result["files"]] == [expected]
        assert result["formats"] == [extension]
    for invalid in ({"formats": [[".pdf"]]}, {"formats": [".exe"]}, {"intent": ""}):
        model.reply = {"queries": [], **invalid}
        with pytest.raises(ModelError):
            expand_query("RAG", model, "test")


def test_extensions_filter_before_limits_and_override_ai_formats(corpus):
    folder, index, model = corpus
    (folder / "最後の資料.MD").write_text("RAGの概要。")
    index.update(folder, model, "embed")
    small = {**DEFAULTS, "candidate_limit": 1}
    for mode in ["keyword", "hybrid"]:
        result = search(index, "RAG", model, small, mode, [".MD"])
        assert [f["relpath"] for f in result["files"]] == ["最後の資料.MD"]
        assert result["formats"] == [".md"]
        assert not search(index, "RAG", model, small, mode, [".pdf"])["files"]
        result = search(index, "RAG", model, DEFAULTS, mode, [".txt", ".md"])
        assert {f["relpath"] for f in result["files"]} == {"青空商事.txt", "最後の資料.MD"}
    model.reply = {"queries": [], "formats": [".pdf"]}
    result = search(index, "RAG", model, DEFAULTS, "expanded", [".md", ".txt"])
    assert result["formats"] == [".md", ".txt"]
    assert len(result["files"]) == 3  # AIの解釈より選択を優先し、ベクトル候補も選択内で取得。
    assert model.last_payload["selected_formats"] == [".md", ".txt"]
    for invalid in ([".exe"], ".pdf", [[".pdf"]]):
        with pytest.raises(ValueError, match="拡張子"):
            search(index, "RAG", model, DEFAULTS, extensions=invalid)


def test_answer_is_scoped_to_one_document(corpus):
    _, index, model = corpus
    file = next(f for f in index.files() if "青空" in f["relpath"])
    result = answer(index, file["id"], "RAGの担当者", model, DEFAULTS)
    assert result["citations"]
    assert "休暇" not in json.dumps(model.last_payload, ensure_ascii=False)


@pytest.mark.parametrize(
    "citation,generated",
    [
        ({"id": 99, "quote": "存在しない原文です"}, "答え[99]"),
        ({"id": 1, "quote": "実際の本文にない内容"}, "答え[1]"),
        ({"id": [], "quote": "引用が不正です"}, "答え[1]"),
        ({"id": 1, "quote": "青空商事でRAGを実装した。"}, "答え[99]"),
        ({"id": 1, "quote": "青空商事でRAGを実装した。"}, "出典番号のない答え"),
    ],
)
def test_invalid_citations_are_not_accepted(corpus, citation, generated):
    _, index, model = corpus
    file = next(f for f in index.files() if "青空" in f["relpath"])
    model.reply = {"supported": True, "answer": generated, "citations": [citation]}
    assert not answer(index, file["id"], "RAGの担当", model, DEFAULTS)["citations"]


def test_quote_whitespace_is_recovered_without_changing_numbers():
    original = "問い合わせ対応時間は平均20分から8分へ短縮した。"
    spaced = "問い合わせ対応時間は平均 20 分から 8 分へ短縮した。"
    assert original_quote(original, spaced) == original
    assert original_quote(original, spaced.replace("8 分", "80 分")) is None
    assert original_quote("開発費180万円\n検証費24万円\n合計204万円", "開発費180万円\n合計204万円") is None


def test_separated_quote_lines_are_verified_individually(corpus):
    _, index, model = corpus
    file = next(f for f in index.files() if "青空" in f["relpath"])
    quote = "青空商事でRAGを実装した。\n費用は120万円。"
    model.reply = {
        "supported": True,
        "answer": "RAGを実装し、費用は120万円です[1]。",
        "citations": [{"id": 1, "quote": quote}],
    }
    result = answer(index, file["id"], "RAGの費用", model, DEFAULTS)
    assert [c["quote"] for c in result["citations"]] == quote.splitlines()
    model.reply["citations"][0]["quote"] = quote.replace("120万円", "1200万円")
    assert not answer(index, file["id"], "RAGの費用", model, DEFAULTS)["citations"]


def test_changed_original_blocks_answer(corpus):
    folder, index, model = corpus
    file = index.files()[0]
    (folder / file["relpath"]).write_text("変わりました")
    with pytest.raises(ValueError, match="原本が変更"):
        answer(index, file["id"], "RAG", model, DEFAULTS)


def test_ollama_offline_is_explicit(corpus):
    _, index, model = corpus
    model.fail_after = 0
    result = search(index, "RAG", model, DEFAULTS)
    assert result["files"] and not result["vector_used"] and result["warnings"]


def test_cloud_models_are_rejected_before_request():
    with pytest.raises(ModelError, match="ローカル"):
        Ollama().verify_local("gpt-oss:20b-cloud")


def test_metrics_split_recall_and_ranking(corpus):
    m = metrics(["wrong", "correct"], ["correct", "wrong"], ["correct"], k=1)
    assert m["candidate_recall"] == 1 and m["recall_at_k"] == 0 and m["mrr"] == 0.5
    assert metrics([], [], [], 5)["no_match_correct"] == 1
    _, index, model = corpus
    with pytest.raises(ValueError, match="索引にない"):
        evaluate(index, [{"query": "RAG", "relevant_files": ["missing.pdf"]}], model, DEFAULTS, ["keyword"])
