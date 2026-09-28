#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if ! command -v uv >/dev/null 2>&1; then
  print "uvを先に導入してください。手順: README.md"
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12.14 .venv
fi
uv pip install --python .venv/bin/python -r requirements-app.txt -r requirements-meili-faiss.txt
if ! command -v meilisearch >/dev/null 2>&1; then
  print "Meilisearchも導入してください: brew install meilisearch"
fi
print "準備できました。start.commandで起動できます。"
