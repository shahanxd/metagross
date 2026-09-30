import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2200
for p in sorted(root.rglob("*.py")):
    if p.name == "__init__.py" or "__pycache__" in p.parts:
        continue
    d = ast.get_docstring(ast.parse(p.read_text(encoding="utf-8")))
    print("=====", p.as_posix())
    print((d or "")[:limit])
