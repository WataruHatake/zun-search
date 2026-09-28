"""学習用のファイル検索。標準ライブラリだけで本文を検索する。"""

import argparse
import sys
from pathlib import Path


def load_documents(folder: Path) -> list[dict]:
    """ファイルのパスと本文をセットで読み込む。"""
    documents = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            print(f"読み込みをスキップ: {path} ({error})", file=sys.stderr)
            continue
        documents.append({"path": path, "text": text})
    return documents


def search_documents(documents: list[dict], query: str) -> list[dict]:
    """本文を行ごとに調べ、ファイルごとに最初の一致を返す。"""
    keyword = query.strip().casefold()
    if not keyword:
        return []

    results = []
    for document in documents:
        for line_number, line in enumerate(document["text"].splitlines(), start=1):
            if keyword in line.casefold():
                results.append({
                    "path": document["path"],
                    "line_number": line_number,
                    "excerpt": line.strip(),
                })
                # 一覧に同じファイルが重複しないよう、次のファイルへ進む。
                break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="テキストファイルの本文をキーワード検索します。")
    parser.add_argument("query", help="本文に含まれる言葉")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "samples",
        help="検索するフォルダ（標準: data/samples）",
    )
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("検索する言葉を入力してください。")
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"検索するフォルダが見つかりません: {folder}")

    documents = load_documents(folder)
    results = search_documents(documents, args.query)
    print(f"検索フォルダ: {folder}")
    print(f"読み込んだファイル: {len(documents)}件 / 一致: {len(results)}件")
    if not results:
        print("一致するファイルがありません。本文に含まれる短い言葉で試してください。")
    for result in results:
        relative_path = result["path"].relative_to(folder)
        print(f"\n{relative_path}（{result['line_number']}行目）")
        print(f"  {result['excerpt']}")


if __name__ == "__main__":
    main()
