"""画面と同じ処理をターミナルからも使う。"""

import argparse
import json
import os
import sys
from pathlib import Path

from rag_app.documents import EXTENSIONS
from rag_app.evaluation import evaluate
from rag_app.meili_faiss import MeiliFaissIndex, database_path, settings
from rag_app.models import Ollama
from rag_app.service import MODES, search
from scripts.launch_meili_faiss import app_environment


def main():
    env = app_environment()
    if env.get("DYLD_LIBRARY_PATH") != os.environ.get("DYLD_LIBRARY_PATH"):
        os.execve(sys.executable, [sys.executable, "-m", "rag_app", *sys.argv[1:]], env)
    parser = argparse.ArgumentParser(description="ローカル資料検索")
    parser.add_argument("action", choices=["index", "search", "evaluate", "status"])
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--mode", choices=list(MODES), default="hybrid")
    parser.add_argument("--extensions", nargs="+", choices=sorted(EXTENSIONS))
    parser.add_argument("--keyword-only", action="store_true")
    parser.add_argument("--no-images", action="store_true", help="画像のAI読み取りを省く")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=["keyword", "hybrid"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    config = settings()
    folder = (args.folder or Path(config["folder"])).expanduser().resolve()
    config["folder"] = str(folder)
    ollama = Ollama()
    try:
        with MeiliFaissIndex(database_path(folder)) as index:
            if args.action == "index":
                result = index.update(
                    folder,
                    ollama,
                    "" if args.keyword_only else config["embedding_model"],
                    image_model=config["chat_model"] if config["read_images"] and not args.no_images else "",
                )
            elif args.action == "search":
                result = search(index, args.query, ollama, config, args.mode, extensions=args.extensions)
            elif args.action == "evaluate":
                if not args.dataset:
                    parser.error("--dataset を指定してください。")
                result = evaluate(
                    index, json.loads(args.dataset.read_text()), ollama, config, args.modes, args.k
                )
            else:
                result = index.stats()
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
            print(args.output)
        else:
            print(rendered)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"エラー: {exc}\n")


if __name__ == "__main__":
    main()
