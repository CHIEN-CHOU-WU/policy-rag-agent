"""Run the Streamlit app in the background for scripted recordings.

Shared by record_demo.py and capture_portfolio.py.
"""

import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def running_app(port: int, env: dict[str, str] | None = None) -> Iterator[str]:
    """Start the app on ``port`` (light theme), yield its URL, and stop it afterwards.

    ``env`` overrides environment variables for the app, e.g. ``{"LLM_PROVIDER": "groq"}``.
    """
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/policy_rag/app.py",
         "--server.headless", "true", "--server.port", str(port), "--theme.base", "light",
         # Hide the "Deploy" button, which means nothing in a recording.
         "--client.toolbarMode", "minimal"],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://localhost:{port}"
    try:
        _wait_for(url)
        yield url
    finally:
        server.terminate()
        server.wait()


def _wait_for(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/_stcore/health", timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Streamlit did not start on {url}")
