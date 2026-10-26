"""Render docs/portfolio/source/architecture.html to docs/portfolio/05-architecture.png.

Run:  uv run --group demo python scripts/render_architecture.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "portfolio" / "source" / "architecture.html"
OUTPUT = ROOT / "docs" / "portfolio" / "05-architecture.png"
SIZE = {"width": 1600, "height": 1200}


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=SIZE, device_scale_factor=1)
        page.goto(SOURCE.as_uri())
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(OUTPUT), clip={"x": 0, "y": 0, **SIZE})
        browser.close()
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
