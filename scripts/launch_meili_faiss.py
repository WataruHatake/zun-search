"""MeilisearchとZun Searchを起動。自分が起動したサーバーだけ停止する。"""

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_app.config import PROJECT  # noqa: E402
from rag_app.meili_faiss import MEILI_URL, STATE, Meili, master_key  # noqa: E402
from rag_app.models import ModelError, Ollama  # noqa: E402


def app_environment():
    env = os.environ.copy()
    torch = importlib.util.find_spec("torch")
    if sys.platform == "darwin" and torch:
        # macOSのローダーでlibompを一本化し、FAISSとPyTorchの二重初期化を防ぐ。
        libraries = str(Path(torch.origin).parent / "lib")
        paths = [libraries, *filter(None, env.get("DYLD_LIBRARY_PATH", "").split(":"))]
        env["DYLD_LIBRARY_PATH"] = ":".join(dict.fromkeys(paths))
    return env


def main():
    os.chdir(PROJECT)
    client = Meili()
    app_url = "http://127.0.0.1:8502"
    try:
        response = client.session.get(
            app_url + "/_stcore/health", headers={"Authorization": None}, timeout=2, allow_redirects=False
        )
        if response.status_code == 200 and response.text.strip() == "ok":
            client.session.close()
            subprocess.run(["open", app_url], check=True)
            return
    except requests.RequestException:
        pass
    children = []
    try:
        try:
            client.request("GET", "/version")
        except RuntimeError:
            binary = shutil.which("meilisearch")
            if not binary:
                raise RuntimeError("Meilisearchを導入してください: brew install meilisearch") from None
            log = (STATE / "meilisearch.log").open("a")
            server = subprocess.Popen(
                [
                    binary,
                    "--http-addr",
                    MEILI_URL.removeprefix("http://"),
                    "--db-path",
                    str(STATE / "data.ms"),
                    "--env",
                    "production",
                    "--no-analytics",
                    "--max-indexing-memory",
                    "1GiB",
                    "--max-indexing-threads",
                    "2",
                ],
                env={**os.environ, "MEILI_MASTER_KEY": master_key()},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.close()
            children.append(server)
            deadline = time.monotonic() + 30
            while True:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Meilisearchを起動できません。{STATE / 'meilisearch.log'}を確認してください。"
                    )
                try:
                    client.request("GET", "/version")
                    break
                except RuntimeError:
                    time.sleep(0.2)
        ollama = Ollama(timeout=2)
        try:
            ollama.request("/api/version")
        except ModelError:
            bundled = Path("/Applications/Ollama.app/Contents/Resources/ollama")
            binary = str(bundled) if bundled.is_file() else shutil.which("ollama")
            if not binary:
                raise RuntimeError("Ollamaを導入してください。") from None
            with (STATE / "ollama.log").open("a") as log:
                server = subprocess.Popen(
                    [binary, "serve"],
                    env={**os.environ, "OLLAMA_HOST": "127.0.0.1:11434"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            children.append(server)
            deadline = time.monotonic() + 30
            while True:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Ollamaを起動できません。{STATE / 'ollama.log'}を確認してください。")
                try:
                    ollama.request("/api/version")
                    break
                except ModelError:
                    time.sleep(0.2)
        finally:
            ollama.session.close()
        app = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app_meili_faiss.py",
                "--server.port=8502",
                "--server.headless=false",
                "--theme.primaryColor=#507C32",
                "--theme.backgroundColor=#f7f9f5",
            ],
            env=app_environment(),
        )
        children.append(app)
        sys.exit(app.wait())
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
        client.session.close()


if __name__ == "__main__":
    main()
