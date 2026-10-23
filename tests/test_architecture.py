"""Guards the ports-and-adapters boundary: only the Chroma adapter may import chromadb."""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "policy_rag"
CHROMA_ADAPTER = PACKAGE_ROOT / "store" / "chroma_store.py"


def imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_only_the_chroma_adapter_imports_chromadb() -> None:
    offenders = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in PACKAGE_ROOT.rglob("*.py")
        if path != CHROMA_ADAPTER
        and any(module.split(".")[0] == "chromadb" for module in imported_modules(path))
    ]
    assert offenders == []
