"""SQLiteに原文・出典・日本語全文索引・埋め込みを保存する。"""

import hashlib
import json
import sqlite3
import time
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from rag_app.documents import (
    EXTRACTOR_VERSION,
    IMAGE_EXTENSIONS,
    PPTX_EXTRACTOR_VERSION,
    XLSX_EXTRACTOR_VERSION,
    Passage,
    discover,
    extract,
    fingerprint,
    split_passages,
)
from rag_app.models import ModelError

TOKENIZER_VERSION = "2"


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


@lru_cache(maxsize=1)
def tokenizer():
    from sudachipy import dictionary

    return dictionary.Dictionary(dict="small").create()


def terms(text: str) -> list[str]:
    from sudachipy import tokenizer as sudachi

    result = []
    for token in tokenizer().tokenize(normalize(text), sudachi.Tokenizer.SplitMode.A):
        if token.part_of_speech()[0] in {"助詞", "助動詞", "補助記号", "記号", "空白"}:
            continue
        word = normalize(token.normalized_form())
        stop_words = {
            "する",
            "ある",
            "いる",
            "こと",
            "もの",
            "探す",
            "ください",
        }
        if (
            any(char.isalnum() for char in word)
            and word not in stop_words
            and normalize(token.dictionary_form()) not in stop_words
        ):
            result.append(word)
    return result


class Index:
    keyword_limit = 40

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, relpath TEXT NOT NULL,
                fingerprint TEXT NOT NULL, status TEXT NOT NULL, warnings TEXT NOT NULL,
                updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                location TEXT NOT NULL, text TEXT NOT NULL, embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(title, folder, body);
            CREATE TRIGGER IF NOT EXISTS delete_search AFTER DELETE ON chunks
            BEGIN DELETE FROM search WHERE rowid=old.id; END;
        """)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def meta(self, key, default=""):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, str(value)))

    def stats(self):
        return {
            "files": self.db.execute("SELECT count(*) FROM files").fetchone()[0],
            "chunks": self.db.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "vectors": self.db.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[
                0
            ],
            "issues": self.db.execute("SELECT count(*) FROM files WHERE status != 'ok'").fetchone()[0],
            "updated": self.meta("updated"),
            "embedding_model": self.meta("embedding_model"),
        }

    def files(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM files ORDER BY relpath")]

    def chunks(self, file_id=None, ids=None):
        sql = "SELECT c.id,c.file_id,c.location,c.text,f.path,f.relpath,f.status,f.fingerprint FROM chunks c JOIN files f ON f.id=c.file_id"
        if ids is not None:
            ids = list(ids)
            rows = []
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                rows.extend(
                    dict(row)
                    for row in self.db.execute(
                        sql + " WHERE c.id IN (" + ",".join("?" for _ in batch) + ")", batch
                    )
                )
            return rows
        return [
            dict(row)
            for row in self.db.execute(
                sql + (" WHERE f.id=?" if file_id is not None else ""),
                (file_id,) if file_id is not None else (),
            )
        ]

    def update(self, folder, ollama=None, embedding_model="", progress=None, *, image_model=""):
        """本文更新はファイル単位で確定。中断後も未作成ベクトルから再開できる。"""
        started = time.perf_counter()
        folder = Path(folder).expanduser().resolve()
        paths, scan_warnings = discover(folder)  # フォルダ消失時には古い索引を削除しない。
        if self.meta("folder") not in {"", str(folder)}:
            raise ValueError("索引の対象フォルダが一致しません。")
        image_signature = ""
        image_reader = None
        if image_model:
            if ollama is None:
                raise ModelError("画像の読み取りにはOllamaが必要です。")
            # 画像の生成指示・縮小方法を変更した場合はimage-v1も更新する。
            image_signature = ollama.verify_local(image_model) + ":image-v1"
            if "vision" not in ollama.request("/api/show", {"model": image_model}).get("capabilities", []):
                raise ModelError("画像入力に対応したモデルを選んでください。")

            def image_reader(data):
                key = "image:" + hashlib.sha256(image_signature.encode() + data).hexdigest()
                cached = self.meta(key, None)
                if cached is not None:
                    return cached
                if progress:
                    progress(number, len(paths), f"画像の文字・図を読み取り中: {path.name}")
                text = ollama.read_image(image_model, data)
                with self.db:
                    self.set_meta(key, text)
                return text

        report = {
            "added": 0,
            "changed": 0,
            "unchanged": 0,
            "removed": 0,
            "errors": 0,
            "embedded": 0,
            "retokenized": 0,
            "warnings": list(scan_warnings),
        }
        old = {f["path"]: f for f in self.files()}
        seen = {str(p) for p in paths}
        changed_extractor = self.meta("extractor") != EXTRACTOR_VERSION
        changed_xlsx = self.meta("extractor_xlsx") != XLSX_EXTRACTOR_VERSION
        changed_pptx = self.meta("extractor_pptx") != PPTX_EXTRACTOR_VERSION
        changed_images = self.meta("image_signature") != image_signature
        for number, path in enumerate(paths, 1):
            if progress:
                progress(number, len(paths), f"読み取り: {path.name}")
            existing = old.get(str(path))
            try:
                before = fingerprint(path)
                if (
                    existing
                    and existing["fingerprint"] == before
                    and existing["status"] != "error"
                    and not changed_extractor
                    and not (changed_xlsx and path.suffix.lower() == ".xlsx")
                    and not (changed_pptx and path.suffix.lower() == ".pptx")
                    and not (changed_images and path.suffix.lower() in IMAGE_EXTENSIONS | {".xlsx", ".pptx"})
                ):
                    if self.meta("tokenizer") != TOKENIZER_VERSION:
                        # 日本語索引の変更だけなら本文抽出・埋め込みは再実行しない。
                        title_tokens = " ".join(terms(path.stem))
                        folder_tokens = " ".join(terms(str(path.parent.relative_to(folder))))
                        with self.db:
                            for row in self.db.execute(
                                "SELECT id,text FROM chunks WHERE file_id=?", (existing["id"],)
                            ).fetchall():
                                self.db.execute("DELETE FROM search WHERE rowid=?", (row["id"],))
                                self.db.execute(
                                    "INSERT INTO search(rowid,title,folder,body) VALUES (?,?,?,?)",
                                    (row["id"], title_tokens, folder_tokens, " ".join(terms(row["text"]))),
                                )
                        report["retokenized"] += 1
                    report["unchanged"] += 1
                    continue
                document = extract(path, image_reader)
                chunks = split_passages(document.passages) or [Passage("ファイル情報", "")]
                if before != fingerprint(path):
                    raise OSError("読み取り中にファイルが変更されました。もう一度更新してください。")
                status = "error" if document.retry else "warning" if document.warnings else "ok"
                report["errors"] += int(document.retry)
                warnings = document.warnings
            except Exception as exc:
                # 変更済み原本に対して古い本文を表示しない。ファイル名のみ残す。
                before, status, warnings = "", "error", [f"読み取り失敗: {type(exc).__name__}: {exc}"]
                chunks = [Passage("ファイル情報", "")]
                report["errors"] += 1
            with self.db:
                if existing:
                    file_id = existing["id"]
                    self.db.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
                    self.db.execute(
                        "UPDATE files SET relpath=?,fingerprint=?,status=?,warnings=?,updated=? WHERE id=?",
                        (
                            str(path.relative_to(folder)),
                            before,
                            status,
                            json.dumps(warnings, ensure_ascii=False),
                            time.time(),
                            file_id,
                        ),
                    )
                    report["changed"] += 1
                else:
                    cursor = self.db.execute(
                        "INSERT INTO files(path,relpath,fingerprint,status,warnings,updated) VALUES (?,?,?,?,?,?)",
                        (
                            str(path),
                            str(path.relative_to(folder)),
                            before,
                            status,
                            json.dumps(warnings, ensure_ascii=False),
                            time.time(),
                        ),
                    )
                    file_id = cursor.lastrowid
                    report["added"] += 1
                title_tokens = " ".join(terms(path.stem))
                folder_tokens = " ".join(terms(str(path.parent.relative_to(folder))))
                for chunk in chunks:
                    cursor = self.db.execute(
                        "INSERT INTO chunks(file_id,location,text) VALUES (?,?,?)",
                        (file_id, chunk.location, chunk.text),
                    )
                    self.db.execute(
                        "INSERT INTO search(rowid,title,folder,body) VALUES (?,?,?,?)",
                        (cursor.lastrowid, title_tokens, folder_tokens, " ".join(terms(chunk.text))),
                    )
        with self.db:
            # 探索エラー時は見えなくなった資料を削除扱いにしない。
            if not scan_warnings:
                for path, existing in old.items():
                    if path not in seen:
                        self.db.execute("DELETE FROM files WHERE id=?", (existing["id"],))
                        report["removed"] += 1
            self.set_meta("folder", folder)
            self.set_meta("extractor", EXTRACTOR_VERSION)
            self.set_meta("extractor_xlsx", XLSX_EXTRACTOR_VERSION)
            self.set_meta("extractor_pptx", PPTX_EXTRACTOR_VERSION)
            self.set_meta("image_signature", image_signature)
            self.set_meta("tokenizer", TOKENIZER_VERSION)
            self.set_meta("updated", time.time())
        if embedding_model and ollama:
            try:
                signature = ollama.verify_local(embedding_model)
                if self.meta("embedding_signature") != signature:
                    with self.db:
                        self.db.execute("UPDATE chunks SET embedding=NULL")
                        self.set_meta("embedding_signature", signature)
                        self.set_meta("embedding_model", embedding_model)
                pending = self.db.execute(
                    "SELECT c.id,c.text,f.relpath,c.location FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.embedding IS NULL AND c.text != ''"
                ).fetchall()
                for start in range(0, len(pending), 8):
                    batch = pending[start : start + 8]
                    if progress:
                        progress(
                            start, max(len(pending), 1), f"ベクトル検索の準備: {start}/{len(pending)}箇所"
                        )
                    vectors = ollama.embed(
                        embedding_model,
                        [f"{r['relpath'][-500:]}\n{r['location']}\n{r['text']}" for r in batch],
                    )
                    with self.db:
                        for row, vector in zip(batch, vectors):
                            self.db.execute(
                                "UPDATE chunks SET embedding=? WHERE id=?",
                                (vector.astype("<f4").tobytes(), row["id"]),
                            )
                    report["embedded"] += len(batch)
            except ModelError as exc:
                report["warnings"].append(
                    f"ベクトル検索の準備が未完了なのだ。全文検索は利用できるのだ。{exc}"
                )
        report["seconds"] = round(time.perf_counter() - started, 2)
        with self.db:
            self.set_meta("last_update_report", json.dumps(report, ensure_ascii=False))
        return report

    def keyword_terms(self, query):
        return tuple(dict.fromkeys(terms(query)))[: self.keyword_limit]

    def keyword(self, query, limit=100, file_id=None, *, file_ids=None):
        words = self.keyword_terms(query)
        if not words:
            return []
        expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
        sql = """SELECT c.id, c.file_id, bm25(search, 5.0, 2.0, 1.0) AS rank
                 FROM search JOIN chunks c ON c.id=search.rowid
                 WHERE search MATCH ?"""
        args = [expression]
        if file_id:
            sql += " AND c.file_id=?"
            args.append(file_id)
        sql += " ORDER BY rank, c.id"
        hits, counts = [], Counter()
        for row in self.db.execute(sql, args):
            if file_ids is not None and row["file_id"] not in file_ids:
                continue
            if not file_id and counts[row["file_id"]] >= 3:
                continue
            counts[row["file_id"]] += 1
            hits.append((row["id"], -row["rank"]))
            if len(hits) >= limit:
                break
        return hits

    def vector(self, vector, limit=100, file_id=None, min_similarity=0.45, *, file_ids=None):
        sql = "SELECT id,file_id,embedding FROM chunks WHERE embedding IS NOT NULL"
        args = ()
        if file_id:
            sql += " AND file_id=?"
            args = (file_id,)
        scored = []
        cursor = self.db.execute(sql, args)
        # ponytail: 全件走査はO(n)。実資料で遅くなったらFAISS等の索引へ置き換える。
        while rows := cursor.fetchmany(512):
            matrix = np.stack([np.frombuffer(row["embedding"], dtype="<f4") for row in rows])
            if matrix.shape[1] != len(vector):
                raise ModelError("埋め込みの次元が一致しません。資料を再更新してください。")
            similarities = matrix @ vector
            scored.extend(
                (r["id"], r["file_id"], float(s)) for r, s in zip(rows, similarities) if s >= min_similarity
            )
        hits, counts = [], Counter()
        for chunk_id, source_id, score in sorted(scored, key=lambda item: (-item[2], item[0])):
            if file_ids is not None and source_id not in file_ids:
                continue
            if not file_id and counts[source_id] >= 3:
                continue
            counts[source_id] += 1
            hits.append((chunk_id, score))
            if len(hits) >= limit:
                break
        return hits
