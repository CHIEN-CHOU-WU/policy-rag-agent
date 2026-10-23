from fakes import FakePageStore, FakeStore, ScriptedLLM, make_chunk, make_page
from policy_rag.agent import ANSWER_NOW, SEARCH_FIRST, run_agent
from policy_rag.llm.base import AssistantTurn, ToolCall, ToolResults, UserMessage
from policy_rag.rag import NOT_FOUND
from policy_rag.tools import PolicyTools

DENTAL = make_chunk("c1", "What isn't covered", "Most dental care isn't covered.", 58, 58)
ADVANTAGE = make_chunk("c2", "Extra benefits", "Some plans cover dental.", 66, 66)
TOOLS_STORE = FakeStore([DENTAL], by_query={"Medicare Advantage dental": [ADVANTAGE]})


def tools() -> PolicyTools:
    return PolicyTools(TOOLS_STORE, FakePageStore([make_page(66, "All of page 66.")]))


def search(query: str, call_id: str = "s1") -> AssistantTurn:
    return AssistantTurn(text="", tool_calls=[ToolCall(call_id, "search_policy", {"query": query})])


def answer(text: str) -> AssistantTurn:
    return AssistantTurn(text=text)


def test_tool_results_are_sent_back_and_the_loop_stops_at_an_answer() -> None:
    llm = ScriptedLLM([search("dental"), answer("Not covered [p. 58].")])
    result = run_agent("Is dental covered?", llm, tools())

    assert result.text == "Not covered [p. 58]."
    second_call = llm.seen[1]
    assert isinstance(second_call[-1], ToolResults)
    [tool_result] = second_call[-1].results
    assert tool_result.call_id == "s1"
    assert '<excerpt page="58"' in tool_result.content


def test_agent_can_search_again_and_read_a_page() -> None:
    llm = ScriptedLLM(
        [
            search("dental"),
            search("Medicare Advantage dental", call_id="s2"),
            AssistantTurn(text="", tool_calls=[ToolCall("g1", "get_page", {"page_number": 66})]),
            answer("Original Medicare doesn't cover it [p. 58]; some plans do [p. 66]."),
        ]
    )
    result = run_agent("How do I add dental coverage?", llm, tools())

    assert [step.tool for step in result.steps] == ["search_policy", "search_policy", "get_page"]
    assert result.cited_pages == [58, 66]
    assert result.unsupported_pages == []
    assert result.cited_sources == [(58, "What isn't covered"), (66, "Extra benefits")]


def test_citation_of_a_page_no_tool_returned_is_flagged() -> None:
    llm = ScriptedLLM([search("dental"), answer("Not covered [p. 58], see also [p. 79].")])
    result = run_agent("q", llm, tools())
    assert result.unsupported_pages == [79]


def test_answering_before_searching_gets_one_nudge() -> None:
    llm = ScriptedLLM([answer(NOT_FOUND), search("dental"), answer("Not covered [p. 58].")])
    result = run_agent("Is dental covered?", llm, tools())

    assert llm.seen[1][-1] == UserMessage(SEARCH_FIRST)
    assert result.text == "Not covered [p. 58]."


def test_the_nudge_is_sent_only_once() -> None:
    llm = ScriptedLLM([answer(NOT_FOUND), answer(NOT_FOUND)])
    result = run_agent("q", llm, tools())
    assert result.text == NOT_FOUND
    assert result.steps == []


def test_tool_errors_are_reported_to_the_model_and_recorded() -> None:
    bad_page = AssistantTurn(text="", tool_calls=[ToolCall("g1", "get_page", {"page_number": 999})])
    llm = ScriptedLLM([search("dental"), bad_page, answer("Not covered [p. 58].")])
    result = run_agent("q", llm, tools())

    [tool_result] = llm.seen[2][-1].results
    assert tool_result.is_error
    assert result.steps[1].error == "There is no page 999."


def test_last_round_asks_for_an_answer_from_what_was_found() -> None:
    llm = ScriptedLLM([search("query 0", "s0"), search("query 1", "s1"), answer("Not covered [p. 58].")])
    result = run_agent("q", llm, tools(), max_rounds=3)

    assert llm.seen[2][-1] == UserMessage(ANSWER_NOW)
    assert result.completed
    assert result.text == "Not covered [p. 58]."


def test_running_out_of_rounds_is_reported_honestly() -> None:
    llm = ScriptedLLM([search(f"query {n}", call_id=f"s{n}") for n in range(3)])
    result = run_agent("q", llm, tools(), max_rounds=3)

    assert not result.completed
    assert "within 3 steps" in result.text
    assert result.cited_pages == []
    assert not result.is_uncited


def test_uncited_answer_is_flagged() -> None:
    llm = ScriptedLLM([search("dental"), answer("Dental isn't covered.")])
    assert run_agent("q", llm, tools()).is_uncited


def test_not_found_after_searching_is_not_flagged_as_uncited() -> None:
    llm = ScriptedLLM([search("dental"), answer(NOT_FOUND)])
    assert not run_agent("q", llm, tools()).is_uncited


def test_on_step_reports_each_tool_call_as_it_happens() -> None:
    seen = []
    llm = ScriptedLLM([search("dental"), answer("Not covered [p. 58].")])
    result = run_agent("q", llm, tools(), on_step=seen.append)
    assert seen == result.steps
