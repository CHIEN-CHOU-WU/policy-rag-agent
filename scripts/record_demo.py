"""Record docs/demo.gif by driving the Streamlit app in a headless browser.

Starts the app, types a question in agent mode, captures frames while the
agent researches and answers, and assembles them into an animated GIF. The
waiting stretch is played back faster so the GIF stays short.

Setup (once):  uv sync --group demo && uv run playwright install chromium
Run:           uv run --group demo python scripts/record_demo.py

Uses whatever LLM_PROVIDER is configured (Ollama by default) and needs the
index built (`uv run python -m policy_rag.ingest`).
"""

import io
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "demo.gif"
PORT = 8599
URL = f"http://localhost:{PORT}"
QUESTION = "What's the penalty if I sign up for Part B late?"

VIEWPORT = {"width": 1100, "height": 760}
GIF_WIDTH = 880
TYPE_FRAME_EVERY = 4  # characters per typing frame
WAIT_FRAME_SECONDS = 1.0  # how often to capture while the agent works
WAIT_FRAME_MS = 250  # ...and how long each of those frames is shown
ANSWER_TIMEOUT_SECONDS = 180


class Recorder:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.frames: list[tuple[Image.Image, int]] = []

    def capture(self, duration_ms: int) -> None:
        png = self.page.screenshot()
        self.frames.append((Image.open(io.BytesIO(png)).convert("RGB"), duration_ms))


def main() -> int:
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/policy_rag/app.py",
         "--server.headless", "true", "--server.port", str(PORT), "--theme.base", "light"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_server()
        frames = _record()
    finally:
        server.terminate()
        server.wait()

    _save_gif(frames)
    print(f"Saved {OUTPUT.relative_to(ROOT)} ({len(frames)} frames, {OUTPUT.stat().st_size / 1e6:.1f} MB)")
    return 0


def _wait_for_server(timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Streamlit did not start on {URL}")


def _record() -> list[tuple[Image.Image, int]]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(URL)
        page.get_by_text("Try one of these").wait_for()
        recorder = Recorder(page)
        recorder.capture(1800)

        box = page.get_by_placeholder("Ask about coverage, costs, or enrollment…")
        box.click()
        for index, character in enumerate(QUESTION, start=1):
            box.press_sequentially(character)
            if index % TYPE_FRAME_EVERY == 0:
                recorder.capture(120)
        recorder.capture(600)
        box.press("Enter")

        # Speed through the research: frequent captures, each shown briefly.
        sources = page.get_by_text("Sources", exact=True)
        deadline = time.time() + ANSWER_TIMEOUT_SECONDS
        while not sources.is_visible():
            if time.time() > deadline:
                raise RuntimeError("No answer within the timeout")
            recorder.capture(WAIT_FRAME_MS)
            time.sleep(WAIT_FRAME_SECONDS)

        page.wait_for_timeout(500)  # let the final render settle
        recorder.capture(6000)
        browser.close()
        return recorder.frames


def _save_gif(frames: list[tuple[Image.Image, int]]) -> None:
    height = round(VIEWPORT["height"] * GIF_WIDTH / VIEWPORT["width"])
    images = [
        image.resize((GIF_WIDTH, height), Image.Resampling.LANCZOS).quantize(colors=128)
        for image, _ in frames
    ]
    OUTPUT.parent.mkdir(exist_ok=True)
    images[0].save(
        OUTPUT,
        save_all=True,
        append_images=images[1:],
        duration=[duration for _, duration in frames],
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    sys.exit(main())
