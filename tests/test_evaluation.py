from pathlib import Path

import pytest

from fakes import FakeLLM
from policy_rag.evaluation import (
    JUDGE_PROMPT,
    Attempt,
    EvalQuestion,
    load_questions,
    parse_verdicts,
    score,
    summarize,
)
from policy_rag.llm.base import AssistantTurn, LLMClient, LLMError
from policy_rag.rag import NOT_FOUND

FLU = EvalQuestion("How often is a flu shot covered?", [44], "Once per flu season.")
PENALTY = EvalQuestion("Is there a Part B late penalty?", [23], "10% per full year, for life.")
PASSPORT = EvalQuestion("How do I renew my passport?", [])


def attempt(
    question: EvalQuestion,
    answer: str,
    cited: tuple[int, ...] = (),
    unsupported: tuple[int, ...] = (),
    seconds: float = 1.0,
    tool_calls: int = 0,
) -> Attempt:
    return Attempt(question, answer, list(cited), list(unsupported), seconds, tool_calls)


TWO_VERDICTS = '[{"id": 1, "correct": true, "reason": "ok"}, {"id": 2, "correct": false, "reason": "no"}]'


class BrokenJudge(LLMClient):
    def chat(self, system, messages, tools) -> AssistantTurn:
        raise LLMError("rate limited")


@pytest.mark.parametrize(
    "reply",
    [
        TWO_VERDICTS,
        '```json\n[{"id": 2, "correct": false, "reason": "no"}, {"id": 1, "correct": true, "reason": "ok"}]\n```',
        f"Here you go: {TWO_VERDICTS}",
    ],
)
def test_parse_verdicts_matches_items_by_id(reply: str) -> None:
    assert [v.correct for v in parse_verdicts(reply, 2)] == [True, False]


def test_missing_or_malformed_items_count_as_incorrect() -> None:
    reply = '[{"id": 1, "correct": "yes"}, {"id": 3, "correct": true}]'
    verdicts = parse_verdicts(reply, 2)
    assert [v.correct for v in verdicts] == [False, False]
    assert "no verdict" in verdicts[1].reason


@pytest.mark.parametrize("reply", ["Looks right to me.", "[not json]", '{"id": 1, "correct": true}'])
def test_unreadable_reply_marks_every_item_incorrect(reply: str) -> None:
    verdicts = parse_verdicts(reply, 2)
    assert [v.correct for v in verdicts] == [False, False]
    assert "Unreadable" in verdicts[0].reason


def test_one_judge_call_grades_every_answerable_attempt() -> None:
    judge = FakeLLM(TWO_VERDICTS)
    results = score(
        [
            attempt(FLU, "Once per flu season [p. 44].", cited=(44,)),
            attempt(PASSPORT, NOT_FOUND),
            attempt(PENALTY, "There is no penalty [p. 23].", cited=(23,)),
        ],
        judge,
    )

    [(system, prompt)] = judge.calls
    assert system == JUDGE_PROMPT
    assert "Item 1\nQuestion: How often is a flu shot covered?" in prompt
    assert "Item 2\nQuestion: Is there a Part B late penalty?" in prompt
    assert "passport" not in prompt  # unanswerable questions are checked by rule
    assert [r.correct for r in results] == [True, True, False]


def test_citing_only_other_pages_is_a_citation_miss() -> None:
    judge = FakeLLM('[{"id": 1, "correct": true, "reason": "ok"}]')
    [result] = score([attempt(FLU, "Once per season [p. 45].", cited=(45,))], judge)
    assert result.correct
    assert result.cites_expected_page is False


def test_declining_an_answerable_question_is_wrong_without_asking_the_judge() -> None:
    judge = FakeLLM("should not be asked")
    [result] = score([attempt(FLU, NOT_FOUND)], judge)
    assert not result.correct
    assert judge.calls == []


def test_unanswerable_questions_must_be_declined() -> None:
    declined, made_up = score(
        [attempt(PASSPORT, NOT_FOUND), attempt(PASSPORT, "Visit travel.state.gov.")], FakeLLM("")
    )
    assert declined.correct
    assert not made_up.correct
    assert made_up.cites_expected_page is None


def test_batch_size_splits_judge_calls() -> None:
    judge = FakeLLM('[{"id": 1, "correct": true, "reason": "ok"}]')
    results = score(
        [attempt(FLU, "Once a season [p. 44].", cited=(44,)), attempt(PENALTY, "10% a year [p. 23].", cited=(23,))],
        judge,
        batch_size=1,
    )
    assert len(judge.calls) == 2
    assert all("Item 1\n" in prompt and "Item 2" not in prompt for _, prompt in judge.calls)
    assert [r.correct for r in results] == [True, True]


def test_judge_failure_counts_as_incorrect() -> None:
    [result] = score([attempt(FLU, "Once per flu season [p. 44].", cited=(44,))], BrokenJudge())
    assert not result.correct
    assert "Judge failed" in result.reason


def test_summary_metrics() -> None:
    results = score(
        [
            attempt(FLU, "Once per flu season [p. 44].", cited=(44,), seconds=2.0, tool_calls=1),
            attempt(
                FLU, "Twice a year [p. 44] [p. 90].", cited=(44, 90), unsupported=(90,), seconds=4.0, tool_calls=3
            ),
            attempt(PASSPORT, NOT_FOUND, seconds=0.0),
            attempt(PASSPORT, "Visit travel.state.gov.", seconds=2.0),
        ],
        FakeLLM(TWO_VERDICTS),
    )
    summary = summarize(results)

    assert summary.answer_accuracy == 0.5
    assert summary.citation_accuracy == 1.0
    assert summary.unsupported_rate == 0.25
    assert summary.decline_accuracy == 0.5
    assert summary.avg_seconds == 2.0
    assert summary.avg_tool_calls == 1.0
    assert summary.failures == [FLU.question, PASSPORT.question]


def test_question_file_is_well_formed() -> None:
    questions = load_questions(Path(__file__).resolve().parent.parent / "eval" / "questions.yaml")
    answerable = [q for q in questions if q.answerable]
    assert len(answerable) == 15 and len(questions) == 18
    assert all(q.reference for q in answerable)
