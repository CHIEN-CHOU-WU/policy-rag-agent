"""Guards the ports-and-adapters boundary: each vendor SDK is imported by exactly one adapter."""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "policy_rag"

ADAPTER_FOR_SDK = {
    "chromadb": PACKAGE_ROOT / "store" / "chroma_store.py",
    "anthropic": PACKAGE_ROOT / "llm" / "anthropic_client.py",
    "ollama": PACKAGE_ROOT / "llm" / "ollama_client.py",
    "openai": PACKAGE_ROOT / "llm" / "openai_compat_client.py",
}


def imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("sdk", sorted(ADAPTER_FOR_SDK))
def test_only_the_adapter_imports_its_sdk(sdk: str) -> None:
    offenders = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in PACKAGE_ROOT.rglob("*.py")
        if path != ADAPTER_FOR_SDK[sdk]
        and any(module.split(".")[0] == sdk for module in imported_modules(path))
    ]
    assert offenders == []
