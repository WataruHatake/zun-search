"""起動口はMeilisearch＋FAISS版に統一する。画面操作はtest_meili_faissで確認する。"""

import subprocess

from rag_app.config import PROJECT


def test_start_command_launches_current_app(tmp_path):
    (tmp_path / ".venv/bin").mkdir(parents=True)
    python = tmp_path / ".venv/bin/python"
    python.write_text('#!/bin/zsh\nprint -r -- "$*"\n')
    python.chmod(0o755)
    for name in ("start.command", "start_meili_faiss.command"):
        script = tmp_path / name
        script.write_text((PROJECT / name).read_text())
        result = subprocess.check_output(["/bin/zsh", str(script)], text=True)
        assert result.strip() == "scripts/launch_meili_faiss.py"
    assert not (PROJECT / "app.py").exists()


def test_launcher_uses_ollama_server_without_opening_chat(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import pytest
    import requests

    from rag_app.models import ModelError
    from scripts import launch_meili_faiss as launcher

    monkeypatch.setattr(launcher, "STATE", tmp_path)
    monkeypatch.setattr(launcher, "app_environment", lambda: {})
    client = MagicMock()
    client.session.get.side_effect = requests.ConnectionError("UI not started")
    monkeypatch.setattr(launcher, "Meili", lambda: client)
    opened = MagicMock()
    monkeypatch.setattr(launcher.subprocess, "run", opened)
    for running in (True, False):
        ollama = MagicMock()
        ollama.request.side_effect = [None] if running else [ModelError("not started"), None]
        monkeypatch.setattr(launcher, "Ollama", lambda **kwargs: ollama)
        server, app = MagicMock(), MagicMock()
        server.poll.return_value = None
        app.poll.return_value = app.wait.return_value = 0
        spawn = MagicMock(side_effect=[app] if running else [server, app])
        monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
        monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/local/bin/" + name)
        with pytest.raises(SystemExit) as exited:
            launcher.main()
        assert exited.value.code == 0
        opened.assert_not_called()
        assert spawn.call_count == (1 if running else 2)
        if not running:
            assert spawn.call_args_list[0].args[0][-1] == "serve"
            assert spawn.call_args_list[0].kwargs["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
            server.terminate.assert_called_once()
        else:
            server.terminate.assert_not_called()
        ollama.session.close.assert_called_once()
