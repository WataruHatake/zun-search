"""SQLiteに本文を保存し、MeilisearchとFAISSで検索する。"""

import fcntl
import hashlib
import json
import os
import secrets
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import requests

from rag_app.config import DEFAULTS
from rag_app.config import STATE as STORAGE
from rag_app.index import Index
from rag_app.models import ModelError

STATE = STORAGE / "meili-faiss"
MEILI_URL = "http://127.0.0.1:7701"


def settings():
    path = STATE / "settings.json"
    return DEFAULTS | json.loads(path.read_text()) if path.exists() else DEFAULTS.copy()


def save_settings(config):
    STATE.mkdir(parents=True, exist_ok=True)
    temporary = STATE / "settings.tmp"
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    temporary.replace(STATE / "settings.json")


def database_path(folder):
    key = hashlib.sha256(str(folder.expanduser().resolve()).encode()).hexdigest()[:20]
    return STATE / "indexes" / f"{key}.sqlite3"


def master_key():
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / "master.key"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text().strip()
    with os.fdopen(descriptor, "w") as stream:
        key = secrets.token_urlsafe(32)
        stream.write(key)
    return key


class Meili:
    def __init__(self):
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers["Authorization"] = "Bearer " + master_key()

    def request(self, method, route, data=None):
        try:
            response = self.session.request(
                method, MEILI_URL + route, json=data, timeout=(3, 120), allow_redirects=False
            )
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"Meilisearch {response.status_code}: {response.text[:300]}")
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                "Meilisearchへ接続できなかったのだ。start.commandで起動してほしいのだ。"
            ) from exc

    def task(self, method, route, data=None):
        task = self.request(method, route, data)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            result = self.request("GET", f"/tasks/{task['taskUid']}")
            if result["status"] == "succeeded":
                return
            if result["status"] in {"failed", "canceled"}:
                raise RuntimeError(f"Meilisearchの索引更新に失敗しました: {result.get('error')}")
            time.sleep(0.1)
        raise RuntimeError(
            "Meilisearchの更新待ちが時間切れになりました。「検索索引を作成・修復」で再開できます。"
        )


class MeiliFaissIndex(Index):
    keyword_limit = 10

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = path.with_suffix(".lock").open("a")
        # ponytail: フォルダ単位で直列実行。複数人が同時利用するなら読取・更新ロックを分ける。
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise RuntimeError(
                "このフォルダは別の処理で使用中です。完了後にもう一度操作してください。"
            ) from exc
        try:
            super().__init__(path)
        except Exception:
            self.lock.close()
            raise
        self.path = path
        self.uid = "ragmf_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]
        self.faiss_path = path.with_suffix(".faiss")
        self.meili = Meili()
        self.faiss_index = None

    def close(self):
        super().close()
        self.meili.session.close()
        self.lock.close()

    def synchronize(self, progress=None):
        started = time.perf_counter()
        with self.db:
            self.set_meta("mf_ready", "0")
        self.faiss_index = None
        # 保存済みの日本語の分かち書きを再利用する。
        rows = self.db.execute(
            "SELECT c.id,c.file_id,s.title,s.folder,s.body,f.path FROM chunks c "
            "JOIN search s ON s.rowid=c.id JOIN files f ON f.id=c.file_id"
        ).fetchall()
        route = f"/indexes/{self.uid}"
        # 設定の更新で索引がなければ作成される。削除も完了を待ってから本文を追加する。
        self.meili.task(
            "PATCH",
            route + "/settings",
            {
                "searchableAttributes": ["title", "folder", "format", "body"],
                "filterableAttributes": ["file_id"],
                "typoTolerance": {"enabled": False},
                "pagination": {"maxTotalHits": max(1000, len(rows))},
            },
        )
        self.meili.task("DELETE", route + "/documents")
        for start in range(0, len(rows), 300):
            if progress:
                progress(start, len(rows), "Meilisearchの全文索引を作成しています…")
            documents = []
            for row in rows[start : start + 300]:
                document = dict(row)
                document["format"] = Path(document.pop("path")).suffix.lower().lstrip(".")
                documents.append(document)
            self.meili.task(
                "POST",
                route + "/documents?primaryKey=id",
                documents,
            )
        vector_index = None
        cursor = self.db.execute("SELECT id,embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id")
        while batch := cursor.fetchmany(512):
            vectors = np.stack([np.frombuffer(row["embedding"], dtype="<f4") for row in batch])
            if not np.isfinite(vectors).all():
                raise ModelError("保存済みベクトルが不正です。資料を更新してください。")
            if vector_index is None:
                # ponytail: まずは正確な全件検索。実測で遅くなったらHNSW等を比較する。
                vector_index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
            if vector_index.d != vectors.shape[1]:
                raise ModelError("異なる次元のベクトルが混在しています。資料を更新してください。")
            vector_index.add_with_ids(vectors, np.array([r["id"] for r in batch], dtype=np.int64))
        if vector_index is None:
            vector_index = faiss.IndexIDMap2(faiss.IndexFlatIP(1))
        temporary = self.faiss_path.with_suffix(".faiss.tmp")
        faiss.write_index(vector_index, str(temporary))
        temporary.replace(self.faiss_path)
        self.faiss_index = vector_index
        report = {
            "meili_chunks": len(rows),
            "faiss_vectors": vector_index.ntotal,
            "seconds": round(time.perf_counter() - started, 2),
        }
        with self.db:
            self.set_meta("mf_ready", "1")
            self.set_meta("mf_sync_report", json.dumps(report))
        return report

    def update(self, folder, ollama=None, embedding_model="", progress=None, *, image_model=""):
        ready = self.meta("mf_ready") == "1" and self.faiss_path.exists()
        signature = self.meta("embedding_signature")
        with self.db:
            self.set_meta("mf_ready", "0")
        before_changes = self.db.total_changes
        try:
            report = super().update(folder, ollama, embedding_model, progress, image_model=image_model)
        except Exception:
            if ready and self.db.total_changes == before_changes:
                with self.db:
                    self.set_meta("mf_ready", "1")
            raise
        changed = any(report[key] for key in ("added", "changed", "removed", "embedded", "retokenized"))
        if changed or not ready or signature != self.meta("embedding_signature"):
            report["search_indexes"] = self.synchronize(progress)
        else:
            with self.db:
                self.set_meta("mf_ready", "1")
        return report

    def _check_ready(self):
        if self.meta("mf_ready") != "1" or not self.faiss_path.exists():
            raise RuntimeError("検索索引の準備が未完了です。「検索索引を作成・修復」を押してください。")

    def _collect(self, hits, limit, file_id, file_ids=None):
        result, counts = [], Counter()
        for chunk_id, source_id, score in hits:
            if file_ids is not None and source_id not in file_ids:
                continue
            if file_id is not None and source_id != file_id:
                continue
            if file_id is None and counts[source_id] >= 3:
                continue
            counts[source_id] += 1
            result.append((chunk_id, score))
            if len(result) >= limit:
                break
        return result

    def keyword(self, query, limit=100, file_id=None, *, file_ids=None):
        self._check_ready()
        words = self.keyword_terms(query)
        if not words or file_ids == set():
            return []

        def hits():
            offset = 0
            while True:
                payload = {
                    "q": " ".join(words),
                    # 全資料の検索は全語一致。選択済み資料への質問は従来どおり候補を広く拾う。
                    "matchingStrategy": "all" if file_id is None else "last",
                    "limit": 500,
                    "offset": offset,
                    "showRankingScore": True,
                    "attributesToRetrieve": ["id", "file_id"],
                }
                if file_id is not None:
                    payload["filter"] = f"file_id = {int(file_id)}"
                elif file_ids is not None:
                    payload["filter"] = "file_id IN " + json.dumps(sorted(file_ids))
                rows = self.meili.request("POST", f"/indexes/{self.uid}/search", payload)["hits"]
                for row in rows:
                    yield row["id"], row["file_id"], row["_rankingScore"]
                if len(rows) < 500:
                    break
                offset += len(rows)

        return self._collect(hits(), limit, file_id, file_ids)

    def vector(self, vector, limit=100, file_id=None, min_similarity=0.45, *, file_ids=None):
        self._check_ready()
        if self.faiss_index is None:
            self.faiss_index = faiss.read_index(str(self.faiss_path))
        if not self.faiss_index.ntotal:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.faiss_index.d or not np.isfinite(query).all():
            raise ModelError("埋め込みの次元または値が不正です。資料を更新してください。")
        radius = float(np.nextafter(np.float32(min_similarity), np.float32(-np.inf)))
        _, scores, ids = self.faiss_index.range_search(query, radius)
        owners = dict(self.db.execute("SELECT id,file_id FROM chunks"))
        hits = sorted(
            ((int(i), owners[int(i)], float(s)) for i, s in zip(ids, scores)),
            key=lambda row: (-row[2], row[0]),
        )
        return self._collect(hits, limit, file_id, file_ids)
