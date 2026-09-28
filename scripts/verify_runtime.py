"""準備済みのサンプルと実モデルで回答・根拠不足を確認する。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_app.config import PROJECT  # noqa: E402
from rag_app.meili_faiss import MeiliFaissIndex, database_path, settings  # noqa: E402
from rag_app.models import Ollama  # noqa: E402
from rag_app.service import answer  # noqa: E402


def main():
    config = settings()
    ollama = Ollama()
    call = ollama.chat_json
    raw = []

    def traced(*args, **kwargs):
        response = call(*args, **kwargs)
        raw.append(response)
        return response

    ollama.chat_json = traced
    results = []
    with MeiliFaissIndex(database_path(PROJECT / "data/demo")) as index:
        source = next(f for f in index.files() if f["relpath"] == "青空商事/RAG_実装報告.pdf")
        for question in [
            "問い合わせ対応時間はどう変わりましたか？",
            "このRAGプロジェクトの来年の売上額はいくらですか？",
        ]:
            print(question, flush=True)
            result = answer(index, source["id"], question, ollama, config)
            results.append({"question": question, "result": result})
    output = {"model": config["chat_model"], "results": results, "raw_model_responses": raw}
    (PROJECT / "output/runtime-qa.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
