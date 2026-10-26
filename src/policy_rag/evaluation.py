"""Score answers end to end: correctness (by a judge model) and citations (by rule).

Correctness needs judgment, because a right answer can be worded many ways, so
a separate judge model compares each answer with a reference answer. Everything
that can be checked mechanically is checked without a model:

- citation accuracy: an answer cites at least one page that holds the answer
- unsupported citations: an answer cites a page it was never shown
- declining: questions the handbook doesn't cover must get the fixed
  "couldn't find this" reply, with no judge involved
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from policy_rag.llm.base import LLMClient, LLMError
from policy_rag.rag import NOT_FOUND


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    pages: list[int]
    reference: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.pages)


def load_questions(path: Path) -> list[EvalQuestion]:
    items = yaml.safe_load(path.read_text())["questions"]
    return [EvalQuestion(item["question"], item["pages"], item.get("reference", "")) for item in items]


@dataclass(frozen=True)
class Attempt:
    """One answer to score, recorded before any judging."""

    question: EvalQuestion
    answer: str
    cited_pages: list[int]
    unsupported_pages: list[int]
    seconds: float
    tool_calls: int = 0

    @property
    def declined(self) -> bool:
        # Leading with the fixed reply counts, even if a short explanation follows.
        return self.answer.strip().startswith(NOT_FOUND)


@dataclass(frozen=True)
class Verdict:
    correct: bool
    reason: str


JUDGE_PROMPT = """\
You grade answers to questions about the "Medicare & You" handbook.

You get numbered items. Each has a question, a reference answer written from the handbook, \
and a candidate answer. Grade each item on its own:
- Correct: the candidate states the reference's key facts and contradicts none of them. \
Extra accurate detail and different wording are fine.
- Incorrect: a key fact is missing or wrong, or the candidate contradicts the reference.
Ignore citation markers such as [p. 12].

Reply with a JSON array only, one object per item:
[{"id": 1, "correct": true or false, "reason": "<one short sentence>"}]"""


def judge_all(llm: LLMClient, attempts: list[Attempt], batch_size: int | None = None) -> list[Verdict]:
    """Grade attempts in batches of ``batch_size`` (all at once by default).

    One call for the whole run keeps a hosted judge inside free-tier quotas
    (Gemini allows 20 requests per model per day). A small local judge does
    better with one item per call, and has no quota to worry about.
    """
    size = batch_size or len(attempts) or 1
    verdicts: list[Verdict] = []
    for start in range(0, len(attempts), size):
        batch = attempts[start : start + size]
        try:
            verdicts.extend(_judge_batch(llm, batch))
        except LLMError as error:
            verdicts.extend([Verdict(False, f"Judge failed: {error}")] * len(batch))
    return verdicts


def _judge_batch(llm: LLMClient, attempts: list[Attempt]) -> list[Verdict]:
    items = "\n\n".join(
        f"Item {number}\n"
        f"Question: {attempt.question.question}\n"
        f"Reference answer: {attempt.question.reference}\n"
        f"Candidate answer: {attempt.answer}"
        for number, attempt in enumerate(attempts, start=1)
    )
    return parse_verdicts(llm.complete(JUDGE_PROMPT, items), len(attempts))


_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def parse_verdicts(reply: str, count: int) -> list[Verdict]:
    """Read the judge's JSON array, tolerating code fences or text around it.

    Anything unreadable or missing counts as incorrect, with the reason saying
    so, so a judge failure can never inflate the score.
    """
    match = _JSON_ARRAY.search(reply)
    try:
        entries = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        entries = None
    if not isinstance(entries, list):
        return [Verdict(False, f"Unreadable judge reply: {reply[:120]!r}")] * count

    by_id: dict[int, Verdict] = {}
    for entry in entries:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("id"), int)
            and isinstance(entry.get("correct"), bool)
        ):
            by_id[entry["id"]] = Verdict(entry["correct"], str(entry.get("reason", "")))
    missing = Verdict(False, "The judge gave no verdict for this item.")
    return [by_id.get(number, missing) for number in range(1, count + 1)]


@dataclass(frozen=True)
class QuestionResult:
    question: str
    answerable: bool
    answer: str
    cited_pages: list[int]
    unsupported_pages: list[int]
    correct: bool
    reason: str
    cites_expected_page: bool | None
    """None for questions the handbook doesn't answer."""
    seconds: float
    tool_calls: int = 0


def score(
    attempts: list[Attempt], judge_llm: LLMClient, batch_size: int | None = None
) -> list[QuestionResult]:
    """Judge the answers that need judgment, and check everything else by rule."""
    needs_judge = [i for i, a in enumerate(attempts) if a.question.answerable and not a.declined]
    verdicts = judge_all(judge_llm, [attempts[i] for i in needs_judge], batch_size)
    judged = dict(zip(needs_judge, verdicts))
    return [_result(attempt, judged.get(index)) for index, attempt in enumerate(attempts)]


def _result(attempt: Attempt, verdict: Verdict | None) -> QuestionResult:
    question = attempt.question
    if not question.answerable:
        correct = attempt.declined
        reason = "Declined as expected." if correct else "Should have declined."
    elif attempt.declined:
        correct, reason = False, "Declined a question the handbook answers."
    else:
        correct, reason = verdict.correct, verdict.reason

    return QuestionResult(
        question=question.question,
        answerable=question.answerable,
        answer=attempt.answer,
        cited_pages=attempt.cited_pages,
        unsupported_pages=attempt.unsupported_pages,
        correct=correct,
        reason=reason,
        cites_expected_page=(
            bool(set(attempt.cited_pages) & set(question.pages)) if question.answerable else None
        ),
        seconds=attempt.seconds,
        tool_calls=attempt.tool_calls,
    )


@dataclass(frozen=True)
class Summary:
    questions: int
    answer_accuracy: float
    """Share of all questions answered correctly (declining counts on unanswerable ones)."""
    citation_accuracy: float
    """Share of answerable questions whose answer cites a page that holds the answer."""
    unsupported_rate: float
    """Share of answers citing at least one page they were never shown."""
    decline_accuracy: float
    """Share of unanswerable questions correctly declined."""
    avg_seconds: float
    avg_tool_calls: float
    failures: list[str] = field(default_factory=list)


def summarize(results: list[QuestionResult]) -> Summary:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    return Summary(
        questions=len(results),
        answer_accuracy=_share(results, lambda r: r.correct),
        citation_accuracy=_share(answerable, lambda r: bool(r.cites_expected_page)),
        unsupported_rate=_share(results, lambda r: bool(r.unsupported_pages)),
        decline_accuracy=_share(unanswerable, lambda r: r.correct),
        avg_seconds=sum(r.seconds for r in results) / len(results) if results else 0.0,
        avg_tool_calls=sum(r.tool_calls for r in results) / len(results) if results else 0.0,
        failures=[r.question for r in results if not r.correct],
    )


def _share(results: list[QuestionResult], test) -> float:
    return sum(1 for r in results if test(r)) / len(results) if results else 0.0
