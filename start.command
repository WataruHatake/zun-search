#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  print "先に ./setup.command を実行してください。"
  exit 1
fi
exec .venv/bin/python scripts/launch_meili_faiss.py
