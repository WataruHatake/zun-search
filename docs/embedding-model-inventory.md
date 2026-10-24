# 検索モデルの棚卸しと比較手順

確認日: **2026-10-24（日本時間）**。コードの棚卸し基準は `804942d3fb20e9d191e995ec1953ba344f612514`。
公式モデルカード・公開記事を調査した仕様比較であり、この環境ではモデルのダウンロード、埋め込み推論、速度計測、日本語資料での品質比較を実行していない。
「設定されているモデル」と「ユーザー端末に実際にインストールされているモデル」は区別する。

## 現在の実装

| 役割 | コード／READMEで確認できるモデル | 実装上の条件 |
| --- | --- | --- |
| 埋め込み | `qwen3-embedding:4b` | `config.py` の初期値。Ollama のローカル `/api/embed`。質問だけに Qwen 用 instruction を付け、出力を L2 正規化する |
| 回答・検索文補完・本文判定 | 初期値 `qwen3.5:4b`、READMEに記録された開発機設定は `qwen3.5:9b` | READMEの保存設定は端末から今回読み出した値ではない。検索モデルの比較中は固定する |
| 専用リランカー | `Qwen/Qwen3-Reranker-0.6B` | `RERANK_MODELS` は 0.6B / 4B / 8B を許可。yes/no logits で採点。MPS または CPU を選択 |
| キーワード検索 | Meilisearch | 埋め込みモデルとは別の対照群として維持 |
| ベクトル検索 | FAISS | モデル出力の正規化済みベクトル。異なるモデルのベクトルを混在させない |

参照: [`config.py`](../rag_app/config.py)、[`models.py`](../rag_app/models.py)、[`index.py`](../rag_app/index.py)、[`README.md`](../README.md)。
現行 `Ollama.embed()` は `num_ctx=4096`、`truncate=False` を指定している。モデル仕様の最大入力長とアプリの実行設定は同一ではない。
画像は `documents.py` / `index.py` で AI 読み取り結果を文字列にしてから索引化する。ページ画像を直接埋め込む経路はない。

## NVIDIAの新モデルはどれか

今回確認した最新の検索用ファミリーは **Nemotron 3.5 Embed / Rerank**。モデルカード上の公開日は **2026-10-20**、NVIDIA執筆の公開記事は **2026-10-21**。
公開記事、現行モデルカード、公式コレクションを照合した範囲では、10月24日までの公開撤回・延期・後継版への置換の告知は確認していない。これは調査範囲内の確認で、将来の更新を保証しない。[S1][S2][S3][S10]

以下は公式記載の仕様。日本語を含む多言語評価があることと、このリポジトリの日本語資料で良いことは別である。

| モデル | 公開日／世代 | パラメーター・標準出力次元・最大入力長 | 入力と日本語 | ライセンス記載／比較上の位置づけ |
| --- | --- | --- | --- | --- |
| Qwen3-Embedding-4B（現行） | 2025年の既存系列 | 4B・2560次元・32K tokens | テキスト、100以上の言語 | Apache-2.0。現行Ollama版を運用基準にする [S4][S5] |
| Nemotron-3-Embed-1B-BF16 | 2026-07-16 | 約1.14B・2048次元・32768 tokens | テキスト、日本語を含む34言語で評価 | OpenMDW-1.1、商用利用可能との記載。軽量なテキスト比較の第一候補 [S6] |
| Nemotron-3-Embed-8B-BF16 | 2026-07-16 | 約8B・4096次元・32768 tokens | テキスト、日本語を含む34言語で評価 | OpenMDW-1.1。計算資源に余裕がある場合の比較候補 [S7] |
| Nemotron-3.5-Embed-1B-BF16 | 2026-10-20 | **実際は1.55B**・2048次元・32768 tokens | 文書はテキスト／画像／両方、質問はテキスト。日本語を含むテキスト34・画像18言語で評価 | OpenMDW-1.1。ページ画像・図表を扱う場合の第一候補 [S2] |
| Nemotron-3.5-Embed-8B-BF16 | 2026-10-20 | 8.38B・4096次元・32768 tokens | 文書はテキスト／画像／両方、質問はテキスト。日本語評価あり | OpenMDW-1.1。画像検索の品質比較候補 [S3] |
| Nemotron-3.5-Embed-1B-NVFP4 | 2026-10-20 | 1.55Bの量子化版・2048次元 | テキスト／画像／両方 | OpenMDW-1.1。公式経路はLinux・vLLM・NVFP4対応GPU（Blackwell）。Mac向け量子化版として扱わない [S8] |
| llama-embed-nemotron-8b | **2025-10-21** | 約7.5B・4096次元・32768 tokens | 多言語テキスト | 非商用／研究専用との記載。2026年10月の新モデルではない [S9] |

重要な違いとして、3.5の言語モデル部分は同サイズの3のものを固定して画像機能を追加している。8Bの公式カードはテキスト埋め込みが同一、1Bは同じテキスト埋め込み空間を維持すると説明している。[S2][S3]
したがって、**現状のテキスト索引を比較するなら Qwen3-Embedding-4B 対 Nemotron-3-Embed-1B-BF16 を優先する**。3.5への名前の変更だけでテキスト精度が向上するとは判断しない。

同時公開の `Nemotron-3.5-Rerank-8B-BF16` は埋め込みモデルではなく、質問と文書を対で入力して関連度のlogitを返す8.38Bの再順位付けモデル（32768 tokens、OpenMDW-1.1）。現行Qwenリランカーの入力・出力処理を流用して名前だけ差し替えることはできない。[S11]
`Nemotron 3.5 Parse` は文書抽出用であり、埋め込み比較とは別に扱う。[S1]

## 公式の比較結果と、このリポジトリの実測を分ける

NVIDIAの2026-10-21記事は、ViDoRe **v3.1** における3.5 Embed 8B BF16の nDCG@10 を61.17、1B BF16を58.38と報告している。これは視覚文書検索の公式報告で、zun-search、現行 `qwen3-embedding:4b`、Mac上の結果ではない。[S1]
v3とv3.1、テキストのみと画像＋テキスト、埋め込みとリランカーで数値を混ぜて順位表にしない。

| 比較事項 | 2026-10-24時点の今回の結果 |
| --- | --- |
| モデル名、公開日、仕様、実装との接続差分 | 上記一次資料・コードから確認 |
| Qwen 4B 対 NVIDIA の日本語 Recall@K / MRR / NDCG@K | **未測定** |
| 回答の正確性、意味的根拠整合性、適切な回答拒否 | **未測定**。回答評価データと人手判定で別途計測 |
| M1 Pro / M4 Pro の速度・メモリ・動作可否 | **未検証**。GPU向け公式サンプルから推定しない |
| モックの引用検証結果 | コードの回帰検証。モデル品質を示すものではない |

## 統合前に必要な変更と確認

1. **埋め込みプロバイダーを分離する。** 現行はOllamaのモデル存在・digest・クラウド拒否を前提としたAPI。NVIDIA公式のTransformers／Sentence Transformers、またはローカルvLLM用アダプターが必要。この変更では未実装であり、上記NVIDIA IDをそのまま設定する手順は提供しない。
2. **モデル固有の前処理を守る。** Nemotron 3は質問の `query: ` と文書の `passage: `、平均poolingとL2正規化。3.5は保存されたchat templateや画像前処理も必要。公式の `encode_query` / `encode_document` を使うと対応するprefixが適用される。Qwen専用instructionを共用しない。[S6][S3]
3. **依存環境を分離する。** 現行 `requirements-reranker.txt` は `transformers>=4.51,<5`。3.5 BF16の公式手順はTransformers 5.20.0系を案内する。1Bの現行カードは画像入力とFlashAttention 2について5.17〜5.19の不具合を記載している。公式vLLM例は `vllm==0.32.0` と `transformers==5.19.0` の別環境であり、両経路の指定を混ぜない。[S2][S3]
4. **索引をモデル別に作る。** モデルID、revision／digest、前処理版、出力次元、正規化、量子化方式、最大入力長をsignatureに含める。現行signatureはOllama digestと `query-v1`。新プロバイダーではこの仕様を拡張し、全文書・質問を同じ空間で処理する。埋め込み空間互換の公式説明がある場合でも、切替時には固定資料で検証する。
5. **画像検索は別実験にする。** ページ画像を索引化するには画像保存、page ID、ページレンダリング、画像と抽出文字の対応、引用位置の管理が必要。画像に対する回答根拠を文字列の部分一致だけで保証しない。

## 比較実験の手順

実測前の設計である。プロバイダー統合後、次の条件を固定して実行する。

1. 同じ資料スナップショット・抽出結果・チャンク・質問集合を使う。日本語、日英混在、固有名詞、表の数値、回答不能、紛らわしい資料を含める。検索用 `relevant_files` と回答用の正解主張・許容根拠は別データとして保持する。
2. **まずテキスト比較:** Meilisearchのみ、現行Qwen 4Bのハイブリッド、Nemotron 3 1Bのハイブリッドを比較。検索文補完・リランカーを固定／無効化して影響を分離する。その後、同一候補集合でQwenリランカーとNVIDIAリランカーを別比較する。
3. **モデル差と運用差を分ける:** 公正なモデル比較では同じ計算機・同等精度・共通入力上限を使う。現行Ollama量子化版とNVIDIA BF16を比較する場合は「構成全体の比較」と明示し、量子化・ハードウェア差をモデル固有の優劣にしない。モデル最大長を使う試験は共通上限試験から分ける。
4. **画像比較:** 同じページに対して「既存AI読み取りテキスト」「画像のみ」「画像＋抽出テキスト」を比べる。回答LLMと検索後のコンテキスト制限を固定する。
5. **検索指標:** 既存 `rag_app/evaluation.py` の candidate recall / Recall@K / MRR / NDCG@K / no-match を使う。警告・埋め込み未使用のfallbackを必ず記録し、正常なベクトル検索の結果に混ぜない。
6. **回答指標:** 同一質問・同一回答LLMで正確性、主張ごとの根拠支持、引用の妥当性、回答不能対応を評価する。引用ID・原文一致・配置検査と、人手の意味判定を別集計する。未判定を0点や合格として補完しない。
7. **時間とメモリ:** コールド起動、ウォーム検索、索引作成を分ける。複数回の中央値／p95、バッチ、コーパス件数、モデル読み込み時間、ピークメモリを記録。FAISSのvector次元による索引サイズも比較する。
8. **閾値:** 現行 `min_similarity=0.45` を新モデルに無条件で移植しない。開発用データで調整し、固定した値を別の評価用データで検証する。検索スコアは意味的な根拠保証ではない。

検索結果JSONと回答結果JSONを別ファイルに保存し、実験IDで関連付ける。付帯情報として確認日・実行日時・実行種別（`mock` / `live`）・データhash・repo commit・モデルrevision・実機・ランタイム版・前処理・エラー件数・人手判定状況を保存する。
未実施の実験には実測スコアを記入しない。比較の結論はまず日本語の保留データと根拠整合性の実測を得てから更新する。

## 出典（すべて2026-10-24確認）

- [S1 NVIDIA執筆記事、2026-10-21](https://huggingface.co/blog/nvidia/nemotron-retriever-world-class-agentic-retrieval)
- [S2 Nemotron-3.5-Embed-1B-BF16 現行公式カード](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16)
- [S3 Nemotron-3.5-Embed-8B-BF16 現行公式カード](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16)
- [S4 Qwen3-Embedding-4B 公式カード](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [S5 Ollama qwen3-embedding:4b](https://ollama.com/library/qwen3-embedding:4b)
- [S6 Nemotron-3-Embed-1B-BF16 現行公式カード](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)
- [S7 Nemotron-3-Embed-8B-BF16 現行公式カード](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16)
- [S8 Nemotron-3.5-Embed-1B-NVFP4 現行公式カード](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-NVFP4)
- [S9 llama-embed-nemotron-8b 公式カード](https://huggingface.co/nvidia/llama-embed-nemotron-8b)
- [S10 NVIDIA公式Retrieverコレクション](https://huggingface.co/collections/nvidia/nemotron-retriever)
- [S11 Nemotron-3.5-Rerank-8B-BF16 現行公式カード](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16)
- [S12 Qwen3-Reranker-0.6B 公式カード](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
