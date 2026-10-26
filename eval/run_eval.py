"""End-to-end evaluation: answer every question in RAG and agent mode, then score.

The answering model comes from LLM_PROVIDER (Ollama by default). Correctness is
graded by a separate judge model:

- ``--judge gemini`` (gemini-3.8-flash) grades a whole run in one call, to stay
  inside the free tier's daily quota (20 requests per model).
- ``--judge ollama`` grades one answer per call with a larger local model;
  free and unlimited, but a weaker and less independent judge.

Answers are saved to eval/results/<provider>-<model>-<mode>-answers.json and
graded results to eval/results/<provider>-<model>-<mode>-judge-<judge>.json. ``--rejudge``
grades the saved answers again (with another judge, or after fixing a
reference answer) without generating them anew, so judges can be compared on
identical answers.

    uv run python eval/run_eval.py                            # both modes, Gemini judge
    uv run python eval/run_eval.py --judge ollama             # local judge (qwen3.6:35b)
    uv run python eval/run_eval.py --rejudge --judge gemini   # grade saved answers again
    uv run python eval/run_eval.py --mode agent               # one mode
    LLM_PROVIDER=groq uv run python eval/run_eval.py          # evaluate another model
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from policy_rag.agent import run_agent
from policy_rag.config import Settings, load_settings
from policy_rag.evaluation import Attempt, EvalQuestion, QuestionResult, load_questions, score, summarize
from policy_rag.llm.base import LLMClient, LLMError
from policy_rag.llm.factory import create_llm_client
from policy_rag.rag import answer_question
from policy_rag.store.factory import create_page_store, create_vector_store
from policy_rag.tools import PolicyTools

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.yaml"
RESULTS_DIR = EVAL_DIR / "results"
MODES = ("rag", "agent")
# The judges behind the README numbers. gemini-3.8-flash was the strictest of the
# judges compared, and its verdicts held up in a manual check.
DEFAULT_GEMINI_JUDGE = "gemini-3.8-flash"
DEFAULT_OLLAMA_JUDGE = "qwen3.6:35b"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate answers and citations end to end.")
    parser.add_argument("--mode", choices=MODES, help="run only one mode (default: both)")
    parser.add_argument("--judge", choices=("gemini", "ollama"), default="gemini", help="judge provider")
    parser.add_argument(
        "--judge-model", help=f"judge model (default: {DEFAULT_GEMINI_JUDGE} or {DEFAULT_OLLAMA_JUDGE})"
    )
    parser.add_argument("--rejudge", action="store_true", help="grade saved answers instead of answering again")
    args = parser.parse_args()

    settings = load_settings()
    tools = PolicyTools(create_vector_store(settings), create_page_store(settings))
    if tools.store.count() == 0:
        print("The index is empty. Run `uv run python -m policy_rag.ingest` first.", file=sys.stderr)
        return 1

    judge_settings = _judge_settings(settings, args.judge, args.judge_model)
    judge_model = _model_name(judge_settings)
    # A hosted judge grades the whole run in one call (quota); a local one, item by item.
    batch_size = None if args.judge == "gemini" else 1
    try:
        llm = create_llm_client(settings)
        judge_llm = create_llm_client(judge_settings)
    except LLMError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    questions = load_questions(QUESTIONS_FILE)
    model = _model_name(settings)
    for mode in [args.mode] if args.mode else MODES:
        print(f"\n=== {settings.llm_provider} ({model}) · {mode} · judged by {args.judge} ({judge_model})")
        run_name = f"{settings.llm_provider}-{_slug(model)}-{mode}"
        answers_path = RESULTS_DIR / f"{run_name}-answers.json"
        if args.rejudge:
            attempts = _load_attempts(answers_path, questions)
        else:
            attempts = []
            for number, question in enumerate(questions, start=1):
                print(f"{number:>2}/{len(questions)} {question.question}", flush=True)
                attempts.append(_attempt(mode, question, llm, tools))
            _save_attempts(answers_path, model, attempts)
        print("Judging...", flush=True)
        results = score(attempts, judge_llm, batch_size)
        for number, result in enumerate(results, start=1):
            mark = "✓" if result.correct else "✗"
            print(f"{number:>2}. {mark} {result.question}  — {result.reason}")
        _report(run_name, settings.llm_provider, model, mode, args.judge, judge_model, results)
    return 0


def _judge_settings(settings: Settings, judge: str, judge_model: str | None) -> Settings:
    if judge == "gemini":
        return replace(settings, llm_provider="gemini", gemini_model=judge_model or DEFAULT_GEMINI_JUDGE)
    return replace(settings, llm_provider="ollama", ollama_model=judge_model or DEFAULT_OLLAMA_JUDGE)


def _attempt(mode: str, question: EvalQuestion, llm: LLMClient, tools: PolicyTools) -> Attempt:
    start = time.perf_counter()
    try:
        if mode == "rag":
            answer = answer_question(question.question, tools.store, llm)
            tool_calls = 0
        else:
            answer = run_agent(question.question, llm, tools)
            tool_calls = len(answer.steps)
        text, cited, unsupported = answer.text, answer.cited_pages, answer.unsupported_pages
    except LLMError as error:
        text, cited, unsupported, tool_calls = f"Error: {error}", [], [], 0
    return Attempt(question, text, cited, unsupported, time.perf_counter() - start, tool_calls)


def _save_attempts(path: Path, model: str, attempts: list[Attempt]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    records = [
        {"question": a.question.question, **{k: v for k, v in asdict(a).items() if k != "question"}}
        for a in attempts
    ]
    path.write_text(json.dumps({"model": model, "answers": records}, ensure_ascii=False, indent=2))


def _load_attempts(path: Path, questions: list[EvalQuestion]) -> list[Attempt]:
    """Saved answers, paired with the current questions file (so fixed references apply)."""
    if not path.exists():
        raise SystemExit(f"No saved answers at {path.relative_to(EVAL_DIR.parent)}; run without --rejudge first.")
    by_text = {question.question: question for question in questions}
    return [
        Attempt(
            question=by_text[record["question"]],
            answer=record["answer"],
            cited_pages=record["cited_pages"],
            unsupported_pages=record["unsupported_pages"],
            seconds=record["seconds"],
            tool_calls=record["tool_calls"],
        )
        for record in json.loads(path.read_text())["answers"]
    ]


def _report(
    run_name: str,
    provider: str,
    model: str,
    mode: str,
    judge: str,
    judge_model: str,
    results: list[QuestionResult],
) -> None:
    summary = summarize(results)
    print(
        f"\nAnswer accuracy    {summary.answer_accuracy:.0%}\n"
        f"Citation accuracy  {summary.citation_accuracy:.0%}  (answerable questions citing a correct page)\n"
        f"Unsupported cites  {summary.unsupported_rate:.0%}  (answers citing a page they never saw)\n"
        f"Declined correctly {summary.decline_accuracy:.0%}  (questions the handbook doesn't cover)\n"
        f"Avg time           {summary.avg_seconds:.1f}s"
        + (f"   Avg tool calls {summary.avg_tool_calls:.1f}" if mode == "agent" else "")
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{run_name}-judge-{judge}.json"
    payload = {
        "provider": provider,
        "model": model,
        "mode": mode,
        "judge": f"{judge} ({judge_model})",
        "summary": asdict(summary),
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved {path.relative_to(EVAL_DIR.parent)}")


def _slug(model: str) -> str:
    """A model name usable in a file name, e.g. "qwen3:8b" -> "qwen3-8b"."""
    return model.replace(":", "-").replace("/", "-")


def _model_name(settings: Settings) -> str:
    return {
        "ollama": settings.ollama_model,
        "gemini": settings.gemini_model,
        "groq": settings.groq_model,
        "anthropic": settings.anthropic_model,
    }[settings.llm_provider]


if __name__ == "__main__":
    sys.exit(main())
