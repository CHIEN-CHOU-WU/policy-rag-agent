"""Render the evaluation results chart for the README and portfolio.

Reads every eval/results/*-judge-gemini.json, draws answer correctness per
model as grouped bars (RAG vs agent), writes the chart as a static HTML page
and screenshots it to a PNG with Playwright.

Setup (once):  uv sync --group demo && uv run playwright install chromium
Run:           uv run --group demo python scripts/render_evaluation_chart.py
"""

import json
import re
from html import escape
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
HTML_OUT = ROOT / "docs" / "portfolio" / "source" / "evaluation.html"
PNG_OUT = ROOT / "docs" / "portfolio" / "06-evaluation.png"

WIDTH, HEIGHT = 1600, 1000

# Display order of the models, grouped by provider; models missing from it go last.
MODEL_ORDER = ["qwen3:8b", "qwen3.6:35b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
PROVIDER_LABELS = {"ollama": "Local · Ollama", "groq": "Groq free tier"}
MODES = [("rag", "RAG"), ("agent", "Agent")]

# Reference palette (dataviz skill), categorical slots 1-2, validated for light mode.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"rag": "#2a78d6", "agent": "#eb6834"}

# Plot geometry.
PLOT_LEFT, PLOT_RIGHT = 150, WIDTH - 70
PLOT_TOP, PLOT_BOTTOM = 330, 780
BAR_WIDTH = 80
BAR_GAP = 8
BAR_RADIUS = 8


def load_results() -> tuple[list[dict], str, int]:
    """Return one row per model with its accuracy per mode, plus judge and question count."""
    models: dict[str, dict] = {}
    judges, question_counts = set(), set()
    for path in sorted(RESULTS.glob("*-judge-gemini.json")):
        data = json.loads(path.read_text())
        row = models.setdefault(data["model"], {"model": data["model"], "provider": data["provider"]})
        row[data["mode"]] = data["summary"]["answer_accuracy"]
        judges.add(data["judge"])
        question_counts.add(data["summary"]["questions"])
    if len(judges) != 1 or len(question_counts) != 1:
        raise SystemExit(f"Results mix judges or question sets: {judges}, {question_counts}")
    for row in models.values():
        missing = [mode for mode, _ in MODES if mode not in row]
        if missing:
            raise SystemExit(f"{row['model']} has no results for {missing}")

    def order(row: dict) -> int:
        return MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else len(MODEL_ORDER)

    judge = judges.pop()
    match = re.search(r"\((.+)\)", judge)  # "gemini (gemini-3.8-flash)" -> "gemini-3.8-flash"
    return sorted(models.values(), key=order), match.group(1) if match else judge, question_counts.pop()


def y_of(value: float) -> float:
    return PLOT_BOTTOM - value * (PLOT_BOTTOM - PLOT_TOP)


def bar_path(x: float, value: float) -> str:
    """A column with a rounded top and a square base on the baseline."""
    top = y_of(value)
    r = min(BAR_RADIUS, (PLOT_BOTTOM - top) / 2)
    right = x + BAR_WIDTH
    return (
        f"M{x},{PLOT_BOTTOM} V{top + r} Q{x},{top} {x + r},{top} "
        f"H{right - r} Q{right},{top} {right},{top + r} V{PLOT_BOTTOM} Z"
    )


def render_svg(rows: list[dict]) -> str:
    parts = []
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(tick)
        color = BASELINE if tick == 0 else GRID
        parts.append(f'<line x1="{PLOT_LEFT}" x2="{PLOT_RIGHT}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text class="tick" x="{PLOT_LEFT - 20}" y="{y + 9}" text-anchor="end">{tick:.0%}</text>')

    band = (PLOT_RIGHT - PLOT_LEFT) / len(rows)
    group_width = len(MODES) * BAR_WIDTH + (len(MODES) - 1) * BAR_GAP
    centers = []
    for i, row in enumerate(rows):
        center = PLOT_LEFT + band * (i + 0.5)
        centers.append(center)
        x = center - group_width / 2
        for mode, label in MODES:
            value = row[mode]
            parts.append(f'<path d="{bar_path(x, value)}" fill="{SERIES[mode]}"><title>{escape(row["model"])} {label}: {value:.0%}</title></path>')
            weight = "700" if value >= 1 else "500"
            parts.append(f'<text class="value" x="{x + BAR_WIDTH / 2}" y="{y_of(value) - 16}" text-anchor="middle" font-weight="{weight}">{value:.0%}</text>')
            x += BAR_WIDTH + BAR_GAP
        name = row["model"].split("/")[-1]
        parts.append(f'<text class="model" x="{center}" y="{PLOT_BOTTOM + 52}" text-anchor="middle">{escape(name)}</text>')

    # Provider brackets under the model names.
    providers: dict[str, list[float]] = {}
    for row, center in zip(rows, centers):
        providers.setdefault(row["provider"], []).append(center)
    y = PLOT_BOTTOM + 82
    for provider, xs in providers.items():
        left, right = min(xs) - band / 2 + 30, max(xs) + band / 2 - 30
        parts.append(
            f'<path d="M{left},{y - 12} V{y} H{right} V{y - 12}" fill="none" stroke="{BASELINE}" stroke-width="2"/>'
        )
        label = escape(PROVIDER_LABELS.get(provider, provider))
        parts.append(f'<text class="provider" x="{(left + right) / 2}" y="{y + 40}" text-anchor="middle">{label}</text>')

    return f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}">{"".join(parts)}</svg>'


def render_html(rows: list[dict], judge: str, questions: int) -> str:
    legend = "".join(
        f'<span class="key"><span class="swatch" style="background:{SERIES[mode]}"></span>{label}</span>'
        for mode, label in MODES
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width={WIDTH}">
<title>Evaluation results</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; margin: 0; }}
  body {{
    width: {WIDTH}px; height: {HEIGHT}px; overflow: hidden;
    background: {SURFACE}; color: {INK};
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  header {{ position: absolute; left: {PLOT_LEFT - 70}px; right: 70px; top: 56px; }}
  h1 {{ font-size: 46px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; }}
  .sub {{ margin-top: 14px; font-size: 30px; color: {INK_SECONDARY}; }}
  .legend {{ display: flex; gap: 40px; margin-top: 26px; font-size: 28px; color: {INK_SECONDARY}; }}
  .key {{ display: inline-flex; align-items: center; gap: 12px; }}
  .swatch {{ width: 26px; height: 26px; border-radius: 6px; }}
  svg {{ position: absolute; left: 0; top: 0; }}
  .tick {{ font-size: 26px; fill: {INK_MUTED}; font-variant-numeric: tabular-nums; }}
  .value {{ font-size: 30px; fill: {INK}; }}
  .model {{ font-size: 32px; font-weight: 600; fill: {INK}; }}
  .provider {{ font-size: 26px; fill: {INK_SECONDARY}; }}
  footer {{ position: absolute; left: {PLOT_LEFT - 70}px; bottom: 40px; font-size: 24px; color: {INK_MUTED}; }}
</style>
</head>
<body>
<header>
  <h1>Free models reach 100% — and the agent isn't always better than RAG</h1>
  <p class="sub">Answer correctness on the Medicare &amp; You 2027 handbook</p>
  <div class="legend">{legend}</div>
</header>
{render_svg(rows)}
<footer>Judged by {escape(judge)} on {questions} questions</footer>
</body>
</html>
"""


def main() -> None:
    rows, judge, questions = load_results()
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(render_html(rows, judge, questions))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(HTML_OUT.as_uri())
        page.screenshot(path=PNG_OUT)
        browser.close()
    for row in rows:
        print(f"{row['model']:<22} RAG {row['rag']:.0%}  Agent {row['agent']:.0%}")
    print(f"Wrote {HTML_OUT.relative_to(ROOT)} and {PNG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
