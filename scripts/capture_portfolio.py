"""Capture the portfolio video and screenshots by driving the web app.

Writes to docs/portfolio/:
  01-demo.mp4           the agent researching and answering a question (cover video)
  02-agent-answer.png   the same answer, with the research steps expanded
  03-rag-answer.png     a RAG-mode answer with its cited sources
  04-citation-check.png an answer whose citation of an unseen page was flagged

The main captures use Groq's qwen/qwen3.8-27b (the best model in the evaluation).
The citation-check capture needs a model that actually cites a page it never
retrieved, so it tries a few question/model pairs that have done so before and
keeps the first real warning; it never fakes one.

Setup (once):  uv sync --group demo && uv run playwright install chromium
Run:           uv run --group demo python scripts/capture_portfolio.py
Needs the index built and GROQ_API_KEY in .env (plus Ollama for the citation check).
"""

import io
import sys
import time

import imageio_ffmpeg
import numpy as np
from PIL import Image
from playwright.sync_api import Browser, Page, sync_playwright

from app_server import ROOT, running_app

OUTPUT = ROOT / "docs" / "portfolio"
PORT = 8598
# A 1200x896 window rendered at 1.5x: 1800px-wide images with sharp text that
# fills the frame (at 1600px wide the centered content left wide empty margins).
WIDTH, HEIGHT = 1200, 896
SCALE = 1.5
# Video encoders want sides divisible by 16; 1792x1344 keeps the 4:3 frame.
VIDEO_SIZE = (1792, 1344)
MIN_HEIGHT = 700
FPS = 10
ANSWER_TIMEOUT_MS = 240_000

GROQ_BEST = {"LLM_PROVIDER": "groq", "GROQ_MODEL": "qwen/qwen3.8-27b"}
AGENT_QUESTION = "How do I add dental coverage?"
RAG_QUESTION = "What's the penalty if I sign up for Part B late?"
# (app environment, mode label, question) pairs that have produced an unsupported citation before.
CITATION_CANDIDATES = [
    ({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen3:8b"}, "RAG — searches once", "When can I join a Medicare drug plan?"),
    ({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen3.6:35b"}, "Agent — researches with tools", "What does Part B cover?"),
    ({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen3:8b"}, "RAG — searches once", "How do I add dental coverage?"),
]
UNSUPPORTED_WARNING = "which its sources never showed"
CHAT_PLACEHOLDER = "Ask about coverage, costs, or enrollment…"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        with running_app(PORT, GROQ_BEST) as url:
            _record_agent(browser, url)
            _capture_rag(browser, url)
        found = _capture_citation_check(browser)
        browser.close()
    if not found:
        print("No unsupported citation occurred in any candidate; 04-citation-check.png was not written.")
    return 0


# --- Captures ----------------------------------------------------------------


def _record_agent(browser: Browser, url: str) -> None:
    """The cover video, then a full screenshot of the finished answer."""
    page = _open(browser, url)
    frames = Frames(page)
    frames.capture(2000)

    box = page.get_by_placeholder(CHAT_PLACEHOLDER)
    box.click()
    for index, character in enumerate(AGENT_QUESTION, start=1):
        box.press_sequentially(character)
        if index % 3 == 0:
            frames.capture(100)
    frames.capture(700)
    box.press("Enter")
    _blur(page)

    # The research happens live; capture it often but play it back at ~5x speed.
    done = page.get_by_text("Sources", exact=True)
    deadline = time.time() + ANSWER_TIMEOUT_MS / 1000
    while not done.is_visible():
        if time.time() > deadline:
            raise RuntimeError("The agent gave no answer in time")
        frames.capture(120)
        time.sleep(0.6)
    page.wait_for_timeout(800)
    frames.capture(2500)

    # Scroll down through the answer to its sources, then hold.
    for _ in range(12):
        page.mouse.wheel(0, 120)
        page.wait_for_timeout(120)
        frames.capture(150)
    frames.capture(4000)
    frames.save_mp4(OUTPUT / "01-demo.mp4")

    page.get_by_text("Research steps (").click()
    page.wait_for_timeout(500)
    _screenshot(page, "02-agent-answer.png")
    page.close()


def _capture_rag(browser: Browser, url: str) -> None:
    page = _open(browser, url)
    page.get_by_text("RAG — searches once").click()
    _ask(page, RAG_QUESTION)
    page.get_by_text("Sources", exact=True).wait_for(timeout=ANSWER_TIMEOUT_MS)
    page.wait_for_timeout(500)
    _screenshot(page, "03-rag-answer.png")
    page.close()


def _capture_citation_check(browser: Browser) -> bool:
    for env, mode, question in CITATION_CANDIDATES:
        with running_app(PORT, env) as url:
            page = _open(browser, url)
            page.get_by_text(mode).click()
            _ask(page, question)
            page.locator("[data-testid=stChatMessage]").nth(1).get_by_text("Sources", exact=True).or_(
                page.get_by_text("cites no pages")
            ).first.wait_for(timeout=ANSWER_TIMEOUT_MS)
            page.wait_for_timeout(500)
            if page.get_by_text(UNSUPPORTED_WARNING).count():
                _screenshot(page, "04-citation-check.png")
                print(f"Citation check captured: {env} · {mode} · {question}")
                page.close()
                return True
            print(f"No unsupported citation: {env} · {mode} · {question}")
            page.close()
    return False


# --- Helpers -----------------------------------------------------------------


def _open(browser: Browser, url: str) -> Page:
    page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=SCALE)
    page.goto(url)
    page.get_by_text("Try one of these").wait_for()
    return page


def _ask(page: Page, question: str) -> None:
    box = page.get_by_placeholder(CHAT_PLACEHOLDER)
    box.fill(question)
    box.press("Enter")


def _screenshot(page: Page, name: str) -> None:
    """Size the window to the whole answer, then capture it from the top."""
    page.mouse.wheel(0, -100_000)
    _blur(page)
    page.wait_for_timeout(300)
    content = page.locator("[data-testid=stMainBlockContainer]").bounding_box()
    # Room below the content for the chat input pinned to the bottom.
    height = max(MIN_HEIGHT, round(content["y"] + content["height"]) + 160) if content else HEIGHT
    page.set_viewport_size({"width": WIDTH, "height": height})
    page.wait_for_timeout(400)
    page.screenshot(path=OUTPUT / name)
    print(f"Saved docs/portfolio/{name} ({round(WIDTH * SCALE)}x{round(height * SCALE)})")


def _blur(page: Page) -> None:
    # A focused chat input gets a red ring that reads like an error.
    page.evaluate("document.activeElement && document.activeElement.blur()")


class Frames:
    """Screenshots, each with how long it stays on screen."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self.frames: list[tuple[Image.Image, int]] = []

    def capture(self, duration_ms: int) -> None:
        image = Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")
        self.frames.append((image, duration_ms))

    def save_mp4(self, path) -> None:
        size = VIDEO_SIZE
        writer = imageio_ffmpeg.write_frames(
            str(path), size, fps=FPS, codec="libx264", pix_fmt_out="yuv420p", quality=8
        )
        writer.send(None)  # start the encoder
        total_ms = 0
        for image, duration_ms in self.frames:
            data = np.asarray(image.resize(size))
            for _ in range(max(1, round(duration_ms * FPS / 1000))):
                writer.send(data.tobytes())
            total_ms += duration_ms
        writer.close()
        print(f"Saved docs/portfolio/{path.name} ({total_ms / 1000:.1f}s, {path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
