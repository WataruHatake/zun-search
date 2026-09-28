# ____ を2か所埋めて、Documentsの場所を表示してください。
from pathlib import Path

documents = Path.home() / "Documents"
print(documents)
