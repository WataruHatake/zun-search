"""引用の原文一致と配置を検査する。主張と根拠の意味的な一致は判定しない。"""

import re

REFERENCE = re.compile(r"\[([0-9]+)\]")
TRAILING_REFERENCES = re.compile(r"(?:\[[0-9]+\]\s*)+[。.!?！？\s\"'」』）)]*$")


def original_quote(text: str, quote: str) -> str | None:
    """空白の差だけを許容し、表示には必ず原文の文字列を返す。"""
    if not isinstance(quote, str):
        return None
    compact_quote = "".join(char for char in quote if not char.isspace())
    if len(compact_quote) < 4:
        return None
    if quote in text:
        return quote
    positions = [position for position, char in enumerate(text) if not char.isspace()]
    compact_text = "".join(text[position] for position in positions)
    start = compact_text.find(compact_quote)
    if start < 0:
        return None
    return text[positions[start] : positions[start + len(compact_quote) - 1] + 1]


def _units(answer):
    """文末と改行による保守的な単位。構文解析による主張の抽出ではない。"""
    units = []
    # 引用は句点の直前・直後のどちらも許す。改行を越えて引用を割り当てない。
    boundary = re.compile(r"[。!?！？]+|\.(?![0-9])")
    after = re.compile(r"[\"'」』）)]*(?:[ \t]*\[[0-9]+\])*[。.!?！？]*")
    for line_number, line in enumerate(answer.splitlines(), 1):
        start = 0
        prefix = re.match(r"\s*[0-9]+[.)]\s+", line)
        scan_start = prefix.end() if prefix else 0
        while start < len(line):
            match = boundary.search(line, max(start, scan_start))
            end = match.end() if match else len(line)
            if match:
                end = after.match(line, end).end()
            text = line[start:end].strip()
            if text:
                units.append({"index": len(units), "line": line_number, "text": text})
            start = end
    return units


def validate_citations(generated, supplied, evidence):
    """Return mechanical checks only; ``valid`` never means semantic faithfulness.

    Plain sentences and list items must each end with one or more ``[id]`` markers.
    Headings, tables, abbreviations and complex quoted punctuation can be flagged
    conservatively. Units and issue codes make these decisions reviewable.
    """
    issues, citations = [], []
    by_id = {entry["id"]: entry for entry in evidence}
    if not isinstance(supplied, list) or not supplied:
        issues.append({"code": "missing_citations"})
    for position, citation in enumerate(supplied if isinstance(supplied, list) else []):
        if not isinstance(citation, dict):
            issues.append({"code": "invalid_citation", "citation_index": position})
            continue
        citation_id = citation.get("id")
        original = by_id.get(citation_id) if type(citation_id) is int else None
        if original is None:
            issues.append({"code": "unknown_citation_id", "citation_index": position})
            continue
        quote = citation.get("quote")
        recovered = original_quote(original["text"], quote)
        quotes = [recovered]
        if recovered is None and isinstance(quote, str) and "\n" in quote:
            # 既存挙動を維持: 離れた原文行を結合した引用は各行を独立に検証・表示。
            quotes = [original_quote(original["text"], line) for line in quote.splitlines() if line.strip()]
        if not quotes or any(part is None for part in quotes):
            issues.append({"code": "quote_not_in_evidence", "citation_index": position, "id": citation_id})
            continue
        citations.extend({**original, "quote": part} for part in quotes)

    units = []
    if not isinstance(generated, str) or not generated.strip():
        issues.append({"code": "empty_or_invalid_answer"})
    else:
        reference_ids = {int(match.group(1)) for match in REFERENCE.finditer(generated)}
        citation_ids = {citation["id"] for citation in citations}
        if reference_ids - citation_ids:
            issues.append({"code": "unverified_reference", "ids": sorted(reference_ids - citation_ids)})
        if citation_ids - reference_ids:
            issues.append({"code": "unused_citation", "ids": sorted(citation_ids - reference_ids)})
        units = _units(generated)
        for unit in units:
            text = unit["text"]
            references = list(REFERENCE.finditer(text))
            unit["citation_ids"] = [int(match.group(1)) for match in references]
            body = REFERENCE.sub("", text)
            # マーカー/句読点/箇条書き記号だけの行は主張への引用とみなさない。
            body = re.sub(r"^\s*(?:[-*+•]|[0-9]+[.)])\s*", "", body)
            has_text = bool(re.search(r"[^\W_]", body, flags=re.UNICODE))
            terminal = TRAILING_REFERENCES.search(text)
            unit["has_terminal_citation"] = bool(terminal and has_text)
            if not has_text:
                issues.append({"code": "orphan_citation_or_empty_unit", "unit": unit["index"]})
            elif not references:
                issues.append({"code": "missing_unit_citation", "unit": unit["index"]})
            elif terminal is None or any(match.start() < terminal.start() for match in references):
                issues.append({"code": "nonterminal_citation", "unit": unit["index"]})
    return {
        "valid": not issues,
        "citations": citations,
        "issues": issues,
        "units": units,
        "method": "quote_match_and_sentence_line_placement_v1",
        "semantic_support": "not_evaluated",
    }
