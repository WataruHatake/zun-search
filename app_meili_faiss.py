"""Zun Search。Meilisearch＋FAISSによる資料検索と出典付き回答。"""

import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

from rag_app.config import PROJECT
from rag_app.documents import EXTENSIONS
from rag_app.evaluation import evaluate
from rag_app.meili_faiss import MeiliFaissIndex, database_path, save_settings, settings
from rag_app.models import RERANK_MODELS, Ollama, prepare_reranker
from rag_app.service import MODES, SEARCH_MODES, answer, search

icon = str(PROJECT / "assets" / "zunda-search.png")
standing = str(PROJECT / "assets" / "zundamon.png")
st.set_page_config(page_title="Zun Search", page_icon=icon, layout="wide")
st.logo(icon, size="large")
st.html("""
<style>
@keyframes zun-sway {
    0%, 100% { transform: rotate(-8deg); }
    50% { transform: translateY(-3px) rotate(8deg); }
}
.st-key-zun_busy img { animation: zun-sway 2.4s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) {
    .st-key-zun_busy img { animation: none; }
}
</style>
""")
with st.container(horizontal=True, vertical_alignment="center"):
    # 表示幅はコンテナで指定し、配信画像の解像度を保つ。
    with st.container(width=96):
        st.image(standing, width="stretch")
    with st.container():
        st.title("Zun Search")
        st.caption("探している資料を、いっしょに見つけるのだ。")
        st.caption(
            "全文検索はMeilisearch、ベクトル検索はFAISSが担当するのだ。"
            "選んだ資料にも、出典付きで質問できるのだ。"
        )
config = settings()
ollama = Ollama()


@contextmanager
def busy(message):
    slot = st.empty()
    with slot.container(key="zun_busy", horizontal=True, vertical_alignment="center"):
        st.image(icon, width=32)
        st.caption(message)
    try:
        yield
    finally:
        slot.empty()


def sync_page(source, target):
    st.session_state[target] = st.session_state[source]


@st.dialog("資料に聞いてみる", width="large", on_dismiss=lambda: st.session_state.pop("selected", None))
def document_dialog(file_id, folder, config):
    try:
        # ダイアログだけの再実行でも使えるよう、毎回DBを開いて閉じる。
        with MeiliFaissIndex(database_path(folder)) as index:
            passages = index.chunks(file_id)
            if not passages:
                st.error("資料が索引に見つからなかったのだ。もう一度検索してほしいのだ。")
                return
            source = passages[0]
            with st.container(horizontal=True, vertical_alignment="center"):
                with st.container(width=64):
                    st.image(standing, width="stretch")
                st.caption("この資料をもとに答えるのだ。知りたいことを聞いてほしいのだ。")
            st.text(source["relpath"])
            open_file, reveal = st.columns(2)
            opened = open_file.button("原本を開く")
            revealed = reveal.button("Finderで表示")
            if opened or revealed:
                path = Path(source["path"]).resolve()
                if path.is_relative_to(folder) and path.is_file():
                    subprocess.run(["open", *(["-R"] if revealed else []), str(path)], check=True)
                else:
                    st.error("原本が見つからなかったのだ。資料を更新してほしいのだ。")
            with st.form(f"question_{file_id}"):
                question = st.text_area("この資料への質問")
                ask = st.form_submit_button("出典付きで回答", type="primary")
            if ask:
                st.session_state.pop("qa", None)
                with busy("資料の根拠を確認しているのだ…"):
                    st.session_state.qa = answer(index, file_id, question, ollama, config)
            st.button(
                "回答をリセット",
                disabled="qa" not in st.session_state,
                on_click=lambda: st.session_state.pop("qa", None),
            )
            if qa := st.session_state.get("qa"):
                st.text(qa["answer"])
                for warning in qa.get("warnings", []):
                    st.warning(warning)
                for citation in qa["citations"]:
                    with st.expander(f"[{citation['id']}] {citation['location']} · 引用", expanded=True):
                        st.text(citation["quote"])
            with st.expander("抽出した本文を確認"):
                for passage in passages:
                    st.caption(passage["location"])
                    st.text(passage["text"])
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        st.error(str(exc))


with st.sidebar:
    st.subheader("検索するフォルダ")
    folder = Path(st.text_input("フォルダの場所", config["folder"])).expanduser().resolve()
    st.caption("探したい資料のフォルダを指定してほしいのだ。更新は変わった資料から進めるのだ。")
    update = st.button("資料を読み込み・更新", type="primary", width="stretch")
    keyword_only = st.checkbox("今回は全文検索だけ準備")
    repair = st.button("検索索引を作成・修復", width="stretch")
    st.caption("修復では、保存済みの本文とベクトルから検索索引を作り直すのだ。")
    st.caption(f"回答：{config['chat_model']}\n\n埋め込み：{config['embedding_model']}")

config["folder"] = str(folder)
context = json.dumps(config, sort_keys=True)
if st.session_state.get("context") != context:
    for key in ("result", "selected", "qa", "evaluation"):
        st.session_state.pop(key, None)
    st.session_state.context = context

try:
    with MeiliFaissIndex(database_path(folder)) as index:
        if update or repair:
            bar = st.progress(0, text="準備しています…")

            def progress(done, total, message):
                bar.progress(min(done / max(total, 1), 1), text=message)

            try:
                with busy("資料を探す準備をしているのだ…"):
                    if repair:
                        report = index.synchronize(progress)
                    else:
                        report = index.update(
                            folder,
                            ollama,
                            "" if keyword_only else config["embedding_model"],
                            progress,
                            image_model=config["chat_model"] if config["read_images"] else "",
                        )
                save_settings(config)
                for key in ("result", "selected", "qa", "evaluation"):
                    st.session_state.pop(key, None)
                st.success("準備できたのだ。資料を探してみてほしいのだ。")
                with st.expander("処理結果"):
                    st.json(report)
                for warning in report.get("warnings", []):
                    st.warning(warning)
            except (ValueError, OSError, RuntimeError) as exc:
                st.error(str(exc))
            finally:
                bar.empty()

        stats = index.stats()
        a, b, c = st.columns(3)
        a.metric("読み込んだ資料", stats["files"])
        b.metric("ベクトル検索の準備", stats["vectors"])
        c.metric("確認が必要な資料", stats["issues"])
        if index.meta("mf_ready") != "1":
            st.info("初回は資料を読み込んでほしいのだ。途中で止まった検索索引は修復できるのだ。")

        search_tab, files_tab, eval_tab, settings_tab = st.tabs(
            ["資料を探す", "読み込み状況", "検索を評価", "設定"]
        )
        with search_tab:
            mode = st.selectbox(
                "検索方式",
                SEARCH_MODES,
                format_func=MODES.get,
                help="全文検索は入力した言葉で、ハイブリッド検索は全文＋ベクトルで探すのだ。"
                "AIによる検索文補完では言い換えを追加し、検索後にLLMが本文を確認するのだ。数十秒〜数分かかるのだ。",
            )
            with st.form("search"):
                query = st.text_input("探している資料")
                extensions = st.multiselect(
                    "拡張子で絞り込む",
                    sorted(EXTENSIONS),
                    key="search_extensions",
                    placeholder="指定なし",
                    help="複数選ぶと、そのどれかに当てはまる資料を探すのだ。"
                    "選択をAIの形式解釈より優先するのだ。未選択なら形式を限定せず、AI補完時は検索文から解釈するのだ。",
                )
                st.caption("検索後の並べ替え")
                use_reranker = False
                if mode == "hybrid":
                    use_reranker = st.checkbox(
                        "専用リランカーで並べ替える",
                        key="use_reranker",
                        help="検索で集めた上位候補を本文の関連度で並べ替えるのだ。追加の処理時間がかかるのだ。"
                        "対象の資料数とモデルは「設定」で変更できるのだ。",
                    )
                elif mode == "expanded":
                    st.caption(
                        "検索文を補完して探したあと、LLMが本文を確認して並べ替え・絞り込みを行うのだ。"
                    )
                else:
                    st.caption("追加の並べ替えは行わず、全文検索の順位を使うのだ。")
                submitted = st.form_submit_button("検索する", type="primary")
            if submitted:
                st.session_state.pop("result", None)
                try:
                    with busy("資料を探しているのだ…"):
                        st.session_state.result = search(
                            index,
                            query,
                            ollama,
                            config,
                            "dedicated" if use_reranker else mode,
                            extensions=extensions,
                        )
                except (ValueError, OSError, RuntimeError) as exc:
                    st.error(str(exc))
            result = st.session_state.get("result")
            if result:
                st.subheader(f"検索結果 · {len(result['files'])} 件")
                st.caption(f"{result['seconds']:.2f}秒 · ファイル単位で表示")
                st.caption(f"{MODES[result['mode']]} · 並べ替え：{result['reranker_used']}")
                with st.expander("検索の内訳"):
                    detail = result["retrieval"]
                    st.caption(
                        f"{MODES[result['mode']]} · 全文の候補 {detail['keyword_chunks']}箇所"
                        f" · ベクトルの候補 {detail['vector_chunks']}箇所"
                    )
                    if result["formats"]:
                        st.caption("拡張子の条件：" + " / ".join(result["formats"]))
                    if result["mode"] == "expanded":
                        st.caption("AIが読み取った意図：" + result["intent"])
                        st.caption(
                            f"候補{len(result['candidate_files'])}資料のうち、上位{len(result['reranked_ids'])}資料の"
                            "本文を確認したのだ。関連が弱い資料と未確認の資料は表示しないのだ。"
                        )
                    elif result["reranked_ids"]:
                        st.caption(
                            f"上位{len(result['reranked_ids'])}資料を並べ替えたのだ。"
                            "対象外の候補は末尾に残しているのだ。"
                        )
                for warning in result["warnings"]:
                    st.warning(warning)
                if result["expansions"]:
                    st.caption("AIが追加した検索表現：" + " / ".join(result["expansions"]))
                per_page = st.selectbox("1ページの表示件数", [10, 20, 50, "すべて"])
                size = (len(result["files"]) or 1) if per_page == "すべて" else per_page
                pages = max(1, (len(result["files"]) + size - 1) // size)
                page_key = f"page_{result['query']}_{','.join(result['formats'])}_{size}_{pages}"
                top_key = page_key + "_top"
                bottom_key = page_key + "_bottom"
                page = st.pagination(pages, key=top_key, on_change=sync_page, args=(top_key, bottom_key))
                for file in result["files"][(page - 1) * size : page * size]:
                    with st.container(border=True):
                        st.text(file["relpath"])
                        passage = file["passages"][0]
                        st.text(passage["text"][:260] or "本文を取得できませんでした。")
                        st.caption(passage["location"] + " · " + " / ".join(file["matches"]))
                        if file.get("rerank_reason"):
                            st.caption("AIの判定：" + file["rerank_reason"])
                        if file["rerank_score"] is not None and file["rerank_score"] < 0.4:
                            st.caption("並べ替えの判定では関連が弱い候補なのだ。")
                        if st.button("この資料に質問する →", key=f"select_{file['file_id']}"):
                            st.session_state.selected = file["file_id"]
                            st.session_state.pop("qa", None)
                st.pagination(pages, key=bottom_key, on_change=sync_page, args=(bottom_key, top_key))
                if not result["files"]:
                    st.info("候補が見つからなかったのだ。言葉を少し変えて探してみてほしいのだ。")

        with files_tab:
            st.dataframe(
                [
                    {
                        "資料": f["relpath"],
                        "状態": f["status"],
                        "確認事項": " / ".join(json.loads(f["warnings"])),
                    }
                    for f in index.files()
                ],
                hide_index=True,
            )
            st.caption("OCR・図の意味理解・Excelの再計算には、まだ対応していないのだ。")
        with eval_tab:
            st.caption("検索文補完や専用リランカーの有無など、同じ問題で検索結果を比べられるのだ。")
            uploaded = st.file_uploader("評価問題のJSON", type="json")
            dataset = json.loads(uploaded.getvalue()) if uploaded else None
            if dataset is None and folder == (PROJECT / "data/demo").resolve():
                dataset = json.loads((PROJECT / "data/evaluation_demo.json").read_text())
            modes = st.multiselect(
                "比較する方式", list(MODES), default=["keyword", "hybrid"], format_func=MODES.get
            )
            k = st.number_input("上位何件を評価するか（k）", 1, 100, 5)
            if st.button("比較を実行", disabled=not dataset or not modes):
                with busy("検索結果を比べているのだ…"):
                    st.session_state.evaluation = evaluate(index, dataset, ollama, config, modes, k)
            if evaluation := st.session_state.get("evaluation"):
                st.dataframe(evaluation["summary"], hide_index=True)
                st.download_button(
                    "比較結果をJSONで保存",
                    json.dumps(evaluation, ensure_ascii=False, indent=2),
                    "meili-faiss-evaluation.json",
                    "application/json",
                )
        with settings_tab:
            with st.form("settings"):
                embedding = st.text_input("ベクトル検索のモデル", config["embedding_model"])
                chat = st.text_input(
                    "回答・検索文補完・画像読み取り・LLM順位付けのモデル", config["chat_model"]
                )
                read_images = st.checkbox(
                    "画像内の文字・図・写真も読み取る",
                    value=config["read_images"],
                    help="上のモデルを使い、画像ファイルとExcel・PowerPointの貼り付け画像を読むのだ。"
                    "初回は画像ごとに時間がかかるのだ。同じ画像の結果は再利用するのだ。"
                    "AIの読み取り・説明には誤りがあるため、出典の画像も確認してほしいのだ。",
                )
                reranker = st.selectbox(
                    "専用リランカー", RERANK_MODELS, index=RERANK_MODELS.index(config["reranker_model"])
                )
                candidates = st.number_input("検索候補の上限（箇所）", 10, 2000, config["candidate_limit"])
                reranks = st.number_input(
                    "本文の確認・並べ替えを行う資料数",
                    1,
                    200,
                    config["rerank_limit"],
                    help="AIによる本文確認と、専用リランカー・LLMによる並べ替えの上限なのだ。",
                )
                similarity = st.slider(
                    "ベクトル検索の類似度の下限",
                    0.0,
                    0.95,
                    float(config["min_similarity"]),
                    0.05,
                    help="ハイブリッド検索・比較・資料への質問に適用するのだ。"
                    "AI補完時は候補を広く取得し、本文の関連性で採否を決めるのだ。",
                )
                if st.form_submit_button("設定を保存"):
                    save_settings(
                        {
                            **config,
                            "embedding_model": embedding.strip(),
                            "chat_model": chat.strip(),
                            "read_images": read_images,
                            "reranker_model": reranker,
                            "candidate_limit": candidates,
                            "rerank_limit": reranks,
                            "min_similarity": similarity,
                        }
                    )
                    st.rerun()
            st.caption(
                "ベクトル検索のモデルを変更したら、資料を更新してほしいのだ。モデルは下で準備できるのだ。"
            )
            with st.expander("モデルを準備する"):
                st.caption(
                    "保存済みの設定にあるモデルを準備するのだ。ダウンロードにはネット接続が必要なのだ。"
                )
                pull_model = st.selectbox(
                    "ダウンロードするOllamaモデル",
                    list(dict.fromkeys([config["embedding_model"], config["chat_model"]])),
                )
                if st.button("モデルをダウンロード"):
                    bar = st.progress(0)
                    try:

                        def pull_progress(data):
                            fraction = data.get("completed", 0) / max(data.get("total", 1), 1)
                            bar.progress(min(fraction, 1), text=data.get("status", "準備中"))

                        ollama.pull(pull_model, pull_progress)
                        st.success("モデルの準備ができたのだ。")
                    finally:
                        bar.empty()
                st.caption("準備する専用リランカー：" + config["reranker_model"])
                if st.button("専用リランカーを準備"):
                    try:
                        with busy("専用リランカーをダウンロードしているのだ…"):
                            prepare_reranker(config["reranker_model"])
                        st.success("専用リランカーの準備ができたのだ。")
                    except Exception as exc:
                        st.error(str(exc))
except (ValueError, OSError, RuntimeError) as exc:
    st.error(str(exc))

if st.session_state.get("selected") is not None:
    document_dialog(st.session_state.selected, folder, config)
