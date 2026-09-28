# Pythonスクリプトで学習するための環境構築

作成日：2026年9月9日

## 1. 今回の指定

- Python：**3.12.14**
- 仮想環境の名前：`.venv`
- 外部ライブラリ：`pdfminer.six==20260107`
- 実行方法：VS Codeのターミナルから `.py` ファイルを実行する。

Python 3.12系で学習環境を統一し、そのセキュリティ更新版である3.12.14を指定する。今後追加するAI関連ライブラリの互換性は、採用・導入時に確認する。

仮想環境は、このプロジェクト専用のPython実行環境とライブラリ置き場。Python本体のバージョンは作成時に指定し、ライブラリはその環境の中へインストールする。

## 2. 作成済みフォルダをVS Codeで開く

VS Codeの「ファイル → フォルダーを開く」で、学習用に作成したフォルダを開く。そのフォルダの直下に `requirements.txt` と `.python-version` を置く。

今回作成したファイルの場所は次のとおり。別のフォルダで学習する場合は、この2ファイルをコピーする。

- [requirements.txt](requirements.txt)
- [.python-version](.python-version)

「ターミナル → 新しいターミナル」を開いて、次を実行する。

```bash
pwd
ls -a
```

`pwd` が学習用フォルダの場所を表示し、`ls -a` に上の2ファイルが含まれていることを確認する。以降のコマンドは、そのフォルダで実行する。新しいプロジェクトフォルダを作る必要はない。

## 3. 指定バージョンのPythonを用意する

この手順では、Pythonのバージョンを指定して導入できる `uv` を使用する。Python 3.12.14はpython.orgでmacOS用インストーラーが提供される通常の保守リリースではないため、uvで用意する。

Homebrewが利用できるターミナルで、以下を順番に実行する。

```bash
brew install uv
uv --version
uv python install 3.12.14
```

すでにuvを使っている場合、`brew install uv` は不要。古いuvで指定バージョンが見つからない場合は、導入元の方法でuvを更新する。Homebrew経由なら `brew upgrade uv` を使う。

`brew: command not found` の場合は、[Homebrew公式サイト](https://brew.sh/)のインストール手順と、完了時に表示されるPATH設定を済ませてから再実行する。

`.python-version` は指定バージョンを記録するファイル。それを置くだけで、現在のターミナルのPythonが自動的に切り替わるわけではない。

## 4. 仮想環境を作る

学習用フォルダで実行する。

```bash
uv venv --python 3.12.14 --seed .venv
```

- `--python 3.12.14`：使うPythonのバージョンを指定する。
- `--seed`：仮想環境にpipも入れる。
- `.venv`：仮想環境の保存先。

このコマンドは初回だけ実行する。すでに `.venv` がある場合は、再作成する前に、次の手順で有効化してバージョンを確認する。

## 5. 仮想環境を有効化する

```bash
source .venv/bin/activate
python --version
python -c "import sys; print(sys.executable)"
```

確認する結果：

- バージョンが `Python 3.12.14`。
- 実行ファイルの場所が、学習用フォルダ内の `.venv/bin/python`。

異なる場合はインストールに進まず、使っている環境を確認する。既存の仮想環境は、有効化するだけではPythonのバージョンが変わらない。

新しいターミナルを開いたときは、必要に応じて `source .venv/bin/activate` を再実行する。

## 6. 必要なライブラリを入れる

仮想環境を有効にしたまま、次を実行する。

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "from pdfminer.high_level import extract_text; print('PDF抽出ライブラリを読み込めました')"
```

`python -m pip` は、現在選択しているPythonのpipを実行する書き方。`-r requirements.txt` は、そのファイルに書かれたライブラリをまとめて入れる指定。

現在のrequirements.txtは次の内容。

```text
pdfminer.six==20260107
```

`==` で直接利用するライブラリのバージョンを固定している。pdfminer.sixが必要とする依存ライブラリはpipが追加で解決・インストールする。このファイルは、間接依存まで固定した完全なロックファイルではない。

`pathlib`、`argparse`、`sqlite3`、`venv` はPythonの標準ライブラリなので、pipで追加する必要はない。

今回の対象は、学習中のPDF抽出・キーワード検索コード。以前の比較用 `read_pdf_spire.py` はSpire.PDF専用であり、このrequirements.txtでは実行対象にしていない。PDF抽出には `from pdfminer.high_level import extract_text` を使った現在の学習コードを保存する。

Meilisearch、FAISS、Ollama連携、リランカー、Docling、pptx・xlsxの読み取り関連は、選定または導入の段階で追加する。Ollama本体やモデルデータは、Pythonのrequirements.txtだけでは導入されない。

## 7. VS Codeで仮想環境を選ぶ

1. VS CodeでPython拡張機能（Microsoft）が有効になっていることを確認する。
2. `Cmd + Shift + P` でコマンドパレットを開く。
3. `Python: Select Interpreter` を選ぶ。
4. 学習用フォルダの `.venv/bin/python` を選ぶ。

一覧に表示されない場合は、インタープリターのパスを入力する操作から `.venv/bin/python` を指定する。

## 8. Notebookのコードを.pyに移して実行する

これまでのコードを、例えば `main.py` に保存する。`main.py` は保存名の例なので、自分が付けた名前に置き換えてよい。

- `import`、関数の定義、関数を使う処理の順にまとめる。
- 別セルで定義していた `read_pdf()` や変数も、実行に必要なものは含める。
- `%pip install ...` など、Notebook専用の行は入れない。インストールはターミナルで行う。
- `pip install pdfminer.six` に対し、コード内のimport名は `pdfminer` になる。

```bash
python main.py
```

`.py` は実行のたびに上から処理される。Notebookで以前のセルに残っていた変数は引き継がれない。

仮想環境の有効化を省略して、次のように実行することもできる。

```bash
.venv/bin/python main.py
```

## 9. 次回の開始と終了

次回はVS Codeで同じフォルダを開き、ターミナルで実行する。

```bash
source .venv/bin/activate
python main.py
```

終了時に環境の有効化を解除する場合：

```bash
deactivate
```

Gitで管理する場合は、`.venv/` と `__pycache__/` を `.gitignore` の対象にする。`requirements.txt` と `.python-version` は共有する設定ファイルとして残す。

## 確認範囲と参考資料

この文書と設定ファイルを作成した時点では、指定の仮想環境の作成・ライブラリのインストールは実施していない。上の手順で、利用する学習フォルダに作成する。

- [Python 3.12.14](https://www.python.org/downloads/release/python-31214/)
- [pdfminer.six 20260107](https://pypi.org/project/pdfminer.six/20260107/)
- [uvのインストール](https://docs.astral.sh/uv/getting-started/installation/)
- [uvでPythonを用意する](https://docs.astral.sh/uv/guides/install-python/)
- [uvの仮想環境](https://docs.astral.sh/uv/pip/environments/)
- [VS CodeのPython環境](https://code.visualstudio.com/docs/python/environments)
