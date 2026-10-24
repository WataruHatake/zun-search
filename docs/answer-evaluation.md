# 回答の根拠整合性を評価する

作成日: 2026-10-24。検索評価は `rag_app/evaluation.py`、回答評価は `rag_app/answer_evaluation.py` に分離する。
回答の評価単位は `service.answer()` と同じ「選択した1資料への質問」。全資料から正しい資料を選べるかは既存の検索指標で別に測る。

## 改善した点と保証範囲

旧処理は有効な引用が回答内のどこかにあれば、別文の引用欠落も通過した。新しい `citations.py` は既存の原文照合に文・改行単位の配置検査を追加する。

| 項目 | 自動処理 | 限界・人手確認 |
| --- | --- | --- |
| 引用ID | 提示抜粋に存在する整数ID、本文の参照との対応を検査 | 同じIDの原文が主張の根拠になるかは別問題 |
| 引用文 | 連続部分の一致を検査。空白差だけ許容し、返す引用は原文から復元 | 抽出文字や画像OCR自体の誤りは原本で確認 |
| 離れた行の引用 | 既存互換のため、改行で結合された引用は各行を個別に検証・表示 | 行間の因果関係や条件を勝手に補ったかは人手確認 |
| 引用配置 | 文・各行の末尾に `[1]` 等を要求。途中だけの引用、引用欠落、単独番号、未使用IDを検出 | 保守的なヒューリスティック。文内の複数主張を完全には分割しない |
| 意味的な支持 | `semantic_support: not_evaluated` | 数字の置換、否定、条件、因果、時点、関係の捏造は原文と照合 |
| 回答不能 | `supported` と正解の回答可能ラベルの一致だけを参考値として保存 | `supported=false` でも本文で捏造できるため、正しい拒否の証明にはしない |

生成形式は平文の短文か箇条書き。句点の前後の引用は許容する。改行を越えて直前文へ引用を割り当てない。見出し・表・英語の略語・複雑な引用符内の句読点は保守的に不整合と判定する場合がある。自然言語の全文法を保証する検査ではない。

`answer()` の既存キー `answer` / `citations` / `warnings` は維持する。追加キーは `supported`、`status`（`answered` / `abstained` / `validation_failed` / `no_evidence`）、`raw_response`、`validation`。`retrieved` は取得できなければ空配列。生成前に根拠が見つからないとき `raw_response` と `validation` は `null`。
`abstained` はモデルの自己申告による分岐、`answered` は機械検査通過を示し、意味的正しさは示さない。

## 正解・回答・レビューを分ける

- `data/answer_evaluation_demo.json`: `kind=answer_grounding`。架空資料の質問・回答可能ラベル・正解回答・必須主張・禁止主張・根拠原文を持つ。実ユーザー資料ではない。
- `data/answer_evaluation_sources/project.md`: 実生成にも使える正解元資料。デモはこのファイルのSHA-256を固定する。
- `data/answer_predictions_mock.json`: `kind=answer_predictions`, `run_type=mock`。手書き応答11件。モデルの実測出力ではない。
- レビューJSON: `kind=answer_reviews`。採点者が入力する意味判定。未記入は `null`。
- レポートJSON: `kind=answer_grounding_report`。自動検査・人手・fixtureの結果を分離。検索のRecall等は含めない。

ケースの必須フィールドは `id`, `question`, `source`, `answerable`, `reference_answer`, `required_claims`, `forbidden_claims`, `evidence`。`source` は評価フォルダ内の相対パス。回答可能なケースは1件以上の必須主張を持ち、各主張の `evidence` に `source_id` と原文に完全一致する `quote` を指定する。回答不能なケースの必須主張は空配列。`source_sha256` を指定すれば実生成前に資料の改変を検出する。

予測JSONには `schema_version: 1`, `kind`, `dataset_sha256`, `run_type`, `metadata`, `records` が必要。各レコードは `case_id`, `response`, `evidence` を持つ。`response` は生成モデルの `supported` / `answer` / `citations` を加工せず保存する。`evidence` はモデルに実際に提示した抜粋であり、正解側の原文を無条件で代入しない。任意の `delivered` はアプリの検証後の出力で、評価対象の生回答と区別する。

モデルの形式違反も評価対象なので、`response` の不正なフィールドは読み込み時に捨てず `response_schema_valid` として報告する。未知・重複・欠損のケースID、不正な原文ID、異なるデータセットへの予測流用は入力エラーにする。
未実行・取得根拠なし・接続失敗は `generation_attempted=false`, `response={}` と実行状態を記録する。これらは生成品質の母数に入れず、レコードは残す。このフラグは評価可能な生成応答を得たかを示し、ネットワーク要求を試みた回数ではない。

## 人手の判定手順

1. READMEの `score` でレポートを作り、正解原文・必須主張・禁止主張・生成回答・実際の提示抜粋を並べて読む。正解データ自体が適切かも、独立した確認者が点検する。
2. `review-template` で空のレビュー票を作る。レビューは **生のresponse** を対象にする。`delivered` の抑止結果をモデルの適切な拒否と取り違えない。
3. 回答の全主張を `claims` に分解し、`text`, `support`, `evidence_ids`, `rationale` を記録する。`support` は `supported` / `unsupported` / `contradicted`。根拠IDは実際に提示された抜粋側のIDを使う。支持される主張は根拠IDが必須。根拠不足は空配列を許す。必要な主張の欠落は `correctness` の理由に記録する。
4. 下表で4軸を判定し、`reviewer` と `rationale` を記入する。`origin=human` は人が実際に読んだレビューだけに使用する。テストの期待判定は `origin=fixture` にし、`run_type=mock` の予測にだけ使用する。未確認の軸を埋めるために推測しない。
5. 同じ予測とレビューで再度 `score`。データセット・回答レコード・予測全体のハッシュが一致しなければ拒否する。回答や質問を修正したら新しいレビュー票を作る。

| 軸 | ラベルと基準 |
| --- | --- |
| `correctness` | `correct`: 必須事項を満たし誤りなし。`partial`: 一部欠落/軽微な不正確さ。`incorrect`: 結論・重要数値・関係の誤り、捏造、回答可能なのに回答しない |
| `grounding` | `supported`: 全主張が提示原文から支持される。`partial`: 支持と無根拠が混在。`unsupported`: 核心が支持されない、または原文と矛盾 |
| `citation_appropriateness` | `appropriate`: 各主張の引用先がその主張を支持。`partial`: 欠落/不適切な引用が一部。`inappropriate`: 核心の引用が無関係/矛盾/欠落 |
| `abstention` | `appropriate`: 回答不能なら具体的な結論を捏造せず限界を伝え、回答可能なら不当に拒否しない。`inappropriate`: 無理な回答または不要な拒否 |

`null` は未確認。`not_applicable` は確認した結果、本当に軸が適用外の場合だけにする。例: 回答不能に適切な拒否のみ返した場合、`correctness=correct`, `abstention=appropriate`, `grounding` / `citation_appropriateness` は `not_applicable` とする。回答可能なのに拒否した場合、`correctness=incorrect`, `abstention=inappropriate`。一部回答可能な質問は、答えられる事項と不明な事項の両方を正解に含めて評価する。

この仕組みは判定票の整合性を検査するが、人が全主張を漏れなく分解したことや判定が正しいことを数学的に検証しない。重要な用途では2名が独立採点し、不一致を原文で解消する。LLM judgeの自動採点機能は今回追加していない。

## 指標と集計

機械検査は原文・ID・配置と応答形式。`declared_answerability_match` はモデルの自己申告との一致にすぎない。意味評価4軸の平均は完全=1、一部=0.5、不適切=0、`null` / `not_applicable` は分母から除く。各軸に採点数・適用数・適用外数・coverageを付け、少数のレビューを全件評価のように表示しない。採点がなければ平均は `null`。

`citation_checks_pass_rate` の集計対象は `supported=true` と申告した応答だけ。拒否を表す文章への引用欠落をこの率の失敗に加えない。ただし `supported=false` の本文にも捏造があり得るため、個別レコードの検査結果と人手確認は残す。各指標の分母はJSONに記録する。

`summary.human` と `summary.fixture` は分離する。モックに人手レビューを付けても `run_type=mock` のままであり、実モデルの品質とはならない。ハッシュは取り違え防止で、入力の真正性や採点者の本人性を証明する署名ではない。

## 回帰ケース

| ケースID | 狙い |
| --- | --- |
| `supported` | 正しい回答と原文引用の対照群 |
| `uncited_extra_claim` | 正しい引用の後に無引用の捏造文を追加。配置検査で検出 |
| `missing_middle_citation` | 引用のある文の間に引用のない文。配置検査で検出 |
| `cited_unsupported_claim` | 同じ文に正しい情報と捏造を混在させ末尾に有効引用。意味レビューが必要 |
| `contradicted_number` | 引用は120万円、回答は1200万円。原文照合だけでは検出不能 |
| `unanswerable_abstention` | 資料にない来年度予算への適切な拒否 |
| `unanswerable_hallucination` | 過去の費用を来年度の予算と誤認 |
| `unnecessary_abstention` | 費用が明記されているのに拒否 |
| `incomplete_answer` | 担当と費用を求められ担当だけ回答 |
| `irrelevant_citation` | 正しい費用に対して休暇手続きの原文を引用 |
| `false_abstention_flag` | `supported=false` でも本文は来年度予算を捏造。自己申告指標の限界 |

これらのモックの合否はコードの回帰検証。実LLMの回答精度・拒否率・NVIDIAモデルの優劣はまだ測定していない。

## 今回の検証記録

2026-10-24、Python 3.12.14で全体テスト **121 passed / 1 skipped**。スキップは起動済みMeilisearchを要する結合テストで、今回は `RAG_TEST_MEILI=1` を指定していない。Ruff検査、変更Pythonファイルの整形確認、`git diff --check` は成功した。既存のSudachi API非推奨警告が1件ある。

`score` と `review-template` をモックJSONで実行し、結果JSON・未記入レビュー票の保存を確認した。サイトパッケージを読み込まない `python -S` でもオフライン採点が動くことをテストした。生の応答が検証で抑止されても保存されること、未生成が拒否成功に加点されないこともモックで確認した。

実Ollama/LLMによる回答生成、NVIDIA埋め込みモデルの推論、日本語品質の比較、Macでの性能計測は未実施。既存のExcelテストは未追跡の個人ファイルに依存していたため、同じassertを保った一時生成fixtureに変更し、公開リポジトリだけで再実行できるようにした。
