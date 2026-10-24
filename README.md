# Zun Search（ずんさーち）

Mac内のPDF・PowerPoint・Excel・画像・Markdown・テキストを検索し、選んだ資料に出典付きで質問するアプリです。全文検索はMeilisearch、ベクトル検索はFAISSが担当します。

## 起動

Finderで **`start.command` をダブルクリック**するか、プロジェクト内で次を実行します。

```sh
./start.command
```

画面は [Zun Search](http://127.0.0.1:8502/) です。Meilisearch（7701）と画面（8502）を起動します。Ollamaは実行中ならそのまま使い、停止中なら `ollama serve` でサーバーだけを起動します。Ollamaのチャット画面は開きません。停止は起動したターミナルでControl+C。自分で起動したサーバーだけを停止します。従来の `start_meili_faiss.command` も同じアプリを起動します。

旧版の画面 `app.py` と8501向け起動処理・旧版へのリンク・保存データ取り込み機能は削除しました。CLIもMeilisearch＋FAISS版に統一しました。現在使用中の `.rag/meili-faiss/`、モデル、原本はそのまま使います。過去の `.rag/indexes/` と `.rag/settings.json` は削除せず残しますが、現在のアプリは参照しません。

## 使い方

1. 左側で対象フォルダを指定し、「資料を読み込み・更新」を押します。変更のない資料は本文抽出・ベクトル化を省き、中断したベクトル化は次の更新で再開します。
2. 検索方式を選び、探している資料を入力します。
3. 必要なら「拡張子で絞り込む」で `.pdf`・`.pptx`・`.xlsx`・`.png`・`.jpg`・`.jpeg`・`.webp`・`.md`・`.txt` を複数選びます。いずれかに該当する資料が対象です。
4. 結果の「この資料に質問する →」から質問できます。出典と原文を確認し、「原本を開く」や「Finderで表示」も利用できます。
5. 原本を追加・変更・移動・削除したら、資料を更新します。

検索結果はファイル単位で表示し、上下のページボタンで移動できます。「回答をリセット」は回答と出典を消し、質問文と選択中の資料は保持します。

Excelはシート・セル位置と列見出しを保ち、割合などの数値表示形式も本文に含めます。数式はファイルに保存された計算結果を使い、未保存なら警告します。タイトル行より下に列見出しがある表の読み取りを改善しました。既存のExcelには次回の資料更新時に適用し、変更のない他形式の資料は再抽出しません。

## 画像とOfficeの図形

| 対象 | 読み取る内容 |
|---|---|
| PNG・JPEG・WebP | 画像内の文字と、図・写真の説明 |
| Excel | セルに加え、図形・テキストボックス内の文字、貼り付け画像、グラフの保存済みデータ |
| PowerPoint | テキストボックス・図形内の文字・グループ・表に加え、貼り付け画像、グラフの保存済みデータ、SmartArt内の文字 |

画像はローカルの画像対応モデルで文字・説明に変換し、その文章をMeilisearch・FAISSで検索します。出典に「画像・AI読み取り」と明示し、図や写真の説明は「AI推定」と区別します。画像内の数字などは原本でも確認してください。画像そのものを直接ベクトル化する方式ではありません。[Ollamaの画像入力](https://docs.ollama.com/capabilities/vision)

「設定」の「画像内の文字・図・写真も読み取る」で有効・無効を切り替えます（初期値はオン）。回答などと同じモデルを使い、現在の `qwen3.5:9b` と初期設定の `qwen3.5:4b` は画像入力に対応します。全文検索だけを準備する場合も、画像読み取りが有効ならこのモデルを使います。

初回は画像ごとに処理時間がかかります。同じ索引内では画像の内容・モデルの版をキーにSQLiteへ結果を保存し、Office内の重複画像やファイル移動後にも再利用します。失敗した画像は次回更新で再試行します。画像設定やモデルを変えたら資料を更新してください。画像・Excel・PowerPointが再抽出の対象になり、変更のないPDF・テキストは再抽出しません。

画像は長辺2048pxに収めて処理します。細かい文字・文字量が多い画像は分割が必要です。4000万画素を超える画像、複数フレームの画像、外部リンク画像、OLEなどの埋め込みファイル本体、旧形式のVML図形・コメント、SmartArtの配置・矢印関係、Officeの図形同士の位置関係には対応していません。PDF内の画像・スキャンPDFのOCRは今回の対象外です。

## 検索方式と並べ替え

| 表示名 | 候補の検索 | 検索後の処理 |
|---|---|---|
| ハイブリッド検索（全文＋ベクトル） | MeilisearchとFAISSの順位をRRFで統合 | 必要なら専用リランカーで並べ替える |
| AIによる検索文補完＋ハイブリッド検索 | AIが意図・資料形式を読み取り、最大2つの補助検索表現を追加 | LLMが上位資料の本文を確認し、並べ替え・絞り込み |
| 全文検索（入力した言葉） | Meilisearchで入力した言葉を検索 | 元の検索順位を表示。モデル不要 |

専用リランカーのチェック項目はハイブリッド検索で表示します（初期値はオフ）。対象外の候補は末尾に残し、失敗した場合は警告付きで元の検索順位を表示します。AI補完ではすでにLLMが本文判定するため、専用リランカーは重ねません。

拡張子の選択は候補の件数制限・並べ替えより前に適用します。明示的な選択をAIの形式解釈より優先します。未選択の場合は形式を限定せず、AI補完時に検索文で形式を指定していればその解釈を使います。候補が見つからなければ0件を表示します。選択を外して再検索すると条件を解除できます。

AIによる本文確認は、設定の資料数（現在20）まで、各資料の取得済み本文を最大3箇所使います。0〜3点中2点以上の資料を表示し、未確認・関連の弱い資料は除きます。数十秒〜数分かかります。解釈の誤りや、候補上限・抜粋以外の根拠の見落としは残ります。補助検索表現、形式の条件、候補数、判定理由、使用モデルを結果画面で確認できます。

## モデルと導入

この開発機はM1 Pro・16GB、本番予定はM4 Pro・48GBです。現在の保存設定は、埋め込み `qwen3-embedding:4b`、回答・検索文補完・本文判定 `qwen3.5:9b`、専用リランカー `Qwen/Qwen3-Reranker-0.6B` です。別環境の初期値は `rag_app/config.py` にあります。大きいモデルの速度と精度は本番機で未検証です。

別のMacではPython 3.12とuv、Ollama、Meilisearchが必要です。

```sh
git clone https://github.com/WataruHatake/zun-search.git
cd zun-search
brew install meilisearch
./setup.command
# 専用リランカーを使う場合
uv pip install --python .venv/bin/python -r requirements-reranker.txt
./start.command
```

画面の「設定」でモデル名を保存し、「モデルを準備する」からOllamaモデル・専用リランカーをダウンロードできます。ダウンロードにはネット接続が必要です。資料の検索・回答はローカルAPIを使います。ベクトル検索のモデルを変更したら資料を更新してください。

学習用スクリプト、`requirements.txt`、`SETUP.md` は履歴として残しています。

NVIDIAの **Nemotron 3.5 Embed / Rerank（2026-10-20公開）** を含む、[検索モデルの棚卸し・仕様比較・実験手順](docs/embedding-model-inventory.md)を追加しました（2026-10-24公式資料確認）。現行Qwenとの日本語品質・Mac速度の比較は未実測です。NVIDIAモデルは前処理・実行環境が異なるため、モデル名だけの差し替えには対応していません。

## CLIと検証

CLIも起動済みのMeilisearchを使い、画面と同じ設定・資料を参照します。

```sh
.venv/bin/python -m rag_app status
.venv/bin/python -m rag_app search "RAG" --mode keyword --extensions .pdf .pptx
.venv/bin/python -m rag_app search "RAG" --mode dedicated
.venv/bin/python -m rag_app index --folder data/demo
# 画像解析・ベクトル化を使わず、本文だけ準備する場合
.venv/bin/python -m rag_app index --folder data/demo --keyword-only --no-images
.venv/bin/python -m rag_app evaluate --folder data/demo \
  --dataset data/evaluation_demo.json --modes keyword hybrid \
  --output output/evaluation.json
```

```sh
uv pip install --python .venv/bin/python -r requirements-test.txt
RAG_TEST_MEILI=1 .venv/bin/pytest -q
.venv/bin/ruff check app_meili_faiss.py rag_app scripts tests
```

Meilisearchの結合テストは一時資料・一時索引を使い、終了時に削除します。`RAG_TEST_MEILI=1` を付けない場合は結合テストをスキップします。

## 回答の根拠評価

検索用の `data/evaluation_demo.json` と Recall@K / MRR / NDCG@K はそのままです。回答評価は別のデータ・CLI・JSONレポートで行います。

`answer()` は引用IDと原文一致に加え、各文・各行の末尾に引用があるか、引用番号だけの行や未使用の引用がないかを検査します。不整合があれば回答を抑止します。引用が正しくても、その引用が主張を裏付けるかは自動保証しません。意味的な正確性・根拠支持・引用の適切さ・回答不能への対応は、正解と原文を使った人手レビューとして別集計します。

次は **手書きのモック応答11件** の検証で、LLMも検索サーバーも起動しません。未レビューの意味指標は `null` です。

```sh
python -m rag_app.answer_eval score \
  --dataset data/answer_evaluation_demo.json \
  --predictions data/answer_predictions_mock.json \
  --output output/answer_mock_report.json

python -m rag_app.answer_eval review-template \
  --dataset data/answer_evaluation_demo.json \
  --predictions data/answer_predictions_mock.json \
  --output output/answer_reviews.json
# 原文・正解と回答を照合し、レビュー票を記入してから実行
python -m rag_app.answer_eval score \
  --dataset data/answer_evaluation_demo.json \
  --predictions data/answer_predictions_mock.json \
  --reviews output/answer_reviews.json \
  --output output/answer_reviewed_report.json
```

実モデルで回答を収集するには、Ollama・Meilisearchを起動し、評価用資料の索引を準備します。

```sh
.venv/bin/python -m rag_app index --folder data/answer_evaluation_sources --no-images
.venv/bin/python -m rag_app.answer_eval run \
  --folder data/answer_evaluation_sources \
  --dataset data/answer_evaluation_demo.json \
  --output output/answer_live_predictions.json
# 上のscore/review-templateで --predictions をこのliveファイルに替える
```

`run` は実際の `service.answer()` を通し、生の生成回答・提示された抜粋・検証後の表示回答・エラーを記録します。生成を止めたケースを適切な回答拒否として自動加点しません。データセットと回答のハッシュで古いレビューの流用を防ぎます。11件は回帰検証用であり、モデルの優劣を結論するには独立した日本語の評価資料が必要です。

[データ形式・判定基準・検査の限界](docs/answer-evaluation.md)を参照してください。今回の変更では実LLMによる品質計測は未実施です。

実装は `app_meili_faiss.py` と `rag_app/` にあります。`index.py` は本文の保存・抽出・差分更新・日本語トークン化の共通基盤とテスト用の検索基準、`meili_faiss.py` は現在の検索エンジンです。SQLiteには原文・出典・ベクトルを保存します。

[詳細な仕様・制限・立ち絵のクレジット・検証履歴](README_MEILI_FAISS.md)も参照してください。Excelの数式再計算には未対応です。回答の引用文と出典IDを検証しますが、画像の読み取りや回答全体の解釈の正しさは引用の存在だけでは保証できません。
