"""機械的検査の回帰試験。実LLMの品質測定ではない。"""

from unittest.mock import Mock, patch

import pytest

from rag_app.citations import validate_citations
from rag_app.config import DEFAULTS
from rag_app.documents import fingerprint
from rag_app.service import answer

EVIDENCE = [
    {"id": 1, "location": "本文", "text": "RAGの導入費は120万円。担当は田中です。"},
    {"id": 2, "location": "別の段落", "text": "検証期間は2週間でした。"},
]
CITATIONS = [{"id": 1, "quote": "RAGの導入費は120万円。"}]


def codes(result):
    return {issue["code"] for issue in result["issues"]}


@pytest.mark.parametrize(
    "generated",
    [
        "導入費は120万円です[1]。",
        "導入費は120万円です。[1]",
        "導入費は120万円です。 [1]",
        "導入費は120万円です[1]",
        "- 導入費は120万円です[1]。\n- 導入費が示されています[1]。",
        "1. 導入費は120万円です[1]。\n2. 導入費が示されています[1]。",
        "The cost is JPY 1.2 million[1].",
    ],
)
def test_sentence_end_markers_remain_compatible(generated):
    result = validate_citations(generated, CITATIONS, EVIDENCE)
    assert result["valid"], result["issues"]
    assert result["semantic_support"] == "not_evaluated"


@pytest.mark.parametrize(
    "generated,code",
    [
        ("導入費は120万円です[1]。売上は倍増しました。", "missing_unit_citation"),
        ("売上は倍増しました。導入費は120万円です[1]。", "missing_unit_citation"),
        ("導入費は120万円です[1]、売上は倍増しました。", "nonterminal_citation"),
        ("- 導入費は120万円です[1]\n- 売上は倍増しました", "missing_unit_citation"),
        ("導入費は120万円です。\n[1]", "orphan_citation_or_empty_unit"),
        ("[1]", "orphan_citation_or_empty_unit"),
        ("[1]導入費は120万円です。", "nonterminal_citation"),
        ("導入費は120万円です[99]。", "unverified_reference"),
        ("The cost is 120 million[1]. Sales doubled.", "missing_unit_citation"),
    ],
)
def test_incomplete_and_misplaced_citations_are_rejected(generated, code):
    result = validate_citations(generated, CITATIONS, EVIDENCE)
    assert not result["valid"]
    assert code in codes(result)


def test_unused_citation_cannot_make_an_answer_appear_better_sourced():
    supplied = CITATIONS + [{"id": 2, "quote": "検証期間は2週間でした。"}]
    result = validate_citations("導入費は120万円です[1]。", supplied, EVIDENCE)
    assert not result["valid"]
    assert "unused_citation" in codes(result)
    assert validate_citations("記載された費用と期間です[1][2]。", supplied, EVIDENCE)["valid"]


@pytest.mark.parametrize(
    "supplied,code",
    [
        (None, "missing_citations"),
        ([], "missing_citations"),
        ([None], "invalid_citation"),
        ([{"id": True, "quote": "RAGの導入費は120万円。"}], "unknown_citation_id"),
        ([{"id": [], "quote": "RAGの導入費は120万円。"}], "unknown_citation_id"),
        ([{"id": 1, "quote": "RAGの導入費は1200万円。"}], "quote_not_in_evidence"),
        ([{"id": 1, "quote": "RAG"}], "quote_not_in_evidence"),
    ],
)
def test_invalid_quotes_and_ids_have_auditable_reasons(supplied, code):
    result = validate_citations("導入費は120万円です[1]。", supplied, EVIDENCE)
    assert not result["valid"]
    assert code in codes(result)


def test_valid_quote_does_not_prove_claim_is_supported():
    # 同じ原文引用を無関係な主張に付ければ形式検査は通る。意味評価で別途判定する。
    result = validate_citations(
        "導入費は120万円です[1]。導入により売上が倍増しました[1]。", CITATIONS, EVIDENCE
    )
    assert result["valid"]
    assert result["semantic_support"] == "not_evaluated"
    assert len(result["units"]) == 2


@pytest.fixture
def mocked_answer(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text(EVIDENCE[0]["text"], encoding="utf-8")
    row = {"path": str(path), "relpath": path.name, "fingerprint": fingerprint(path), **EVIDENCE[0]}
    index, ollama = Mock(), Mock()
    index.chunks.return_value = [row]
    with patch("rag_app.service.retrieve", return_value=([row], [], {})):
        yield index, ollama


def test_service_blocks_uncited_addition_but_retains_raw_output_for_review(mocked_answer):
    index, ollama = mocked_answer
    generated = "導入費は120万円です[1]。導入により売上が倍増しました。"
    ollama.chat_json.return_value = {"supported": True, "answer": generated, "citations": CITATIONS}
    result = answer(index, 1, "RAGの導入費は？", ollama, DEFAULTS)
    assert result["status"] == "validation_failed"
    assert result["supported"] is False
    assert result["citations"] == []
    assert result["raw_response"]["answer"] == generated
    assert "missing_unit_citation" in codes(result["validation"])


def test_service_exposes_semantic_limit_with_mocked_unsupported_claim(mocked_answer):
    index, ollama = mocked_answer
    generated = "導入費は120万円です[1]。導入により売上が倍増しました[1]。"
    ollama.chat_json.return_value = {"supported": True, "answer": generated, "citations": CITATIONS}
    result = answer(index, 1, "RAGの導入費と効果は？", ollama, DEFAULTS)
    assert result["status"] == "answered"
    assert result["answer"] == generated
    assert result["validation"]["semantic_support"] == "not_evaluated"


def test_abstention_is_distinct_from_validation_failure(mocked_answer):
    index, ollama = mocked_answer
    ollama.chat_json.return_value = {"supported": False, "answer": "根拠がありません。", "citations": []}
    result = answer(index, 1, "来年の売上は？", ollama, DEFAULTS)
    assert result["status"] == "abstained"
    assert result["supported"] is False


def test_no_evidence_does_not_invoke_llm(mocked_answer):
    index, ollama = mocked_answer
    with patch("rag_app.service.retrieve", return_value=([], [], {})):
        result = answer(index, 1, "来年の売上は？", ollama, DEFAULTS)
    assert result["status"] == "no_evidence"
    assert result["raw_response"] is None
    assert result["retrieved"] == []
    ollama.chat_json.assert_not_called()
