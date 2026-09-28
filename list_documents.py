"""学習ステップ1: 対象ファイルの名前と保存場所を一覧にする。"""

import argparse
from pathlib import Path


def find_documents(folder: Path) -> list[Path]:
    """指定フォルダの中からPDF・pptx・xlsxのパスを集める。"""
    documents = []

    # rglobは、指定フォルダの下にあるフォルダも調べる。
    for path in folder.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        # Officeが作成する一時ファイルを除外する。
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() in {".pdf", ".pptx", ".xlsx"}:
            documents.append(path)

    return sorted(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="資料の名前と保存場所を一覧にします。")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path.home() / "Documents",
        help="調べるフォルダ（標準: 自分のDocuments）",
    )
    args = parser.parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"フォルダが見つかりません: {folder}")

    try:
        documents = find_documents(folder)
    except OSError as error:
        parser.exit(1, f"フォルダの一覧取得でエラーが発生しました: {error}\n")

    print(f"調べた場所: {folder}")
    print(f"見つかった対象ファイル: {len(documents)}件")
    for path in documents:
        print(f"\nファイル名: {path.name}")
        print(f"保存場所: {path.parent}")


if __name__ == "__main__":
    main()
