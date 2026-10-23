from fakes import FakePageStore, FakeStore, make_chunk, make_page
from policy_rag.llm.base import ToolCall
from policy_rag.tools import TOOL_SPECS, PolicyTools, run_tool

DENTAL = make_chunk("c1", "What isn't covered", "Most dental care isn't covered.", 58, 58)
SIGN_UP = make_chunk("c2", "Signing up", "Sign up around your 65th birthday.", 17, 18)
TOOLS = PolicyTools(FakeStore([DENTAL, SIGN_UP]), FakePageStore([make_page(58, "Full text of page 58.")]))


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="call_1", name=name, arguments=arguments)


def test_tool_specs_have_strict_object_schemas() -> None:
    assert [spec.name for spec in TOOL_SPECS] == ["search_policy", "get_page"]
    for spec in TOOL_SPECS:
        assert spec.input_schema["type"] == "object"
        assert spec.input_schema["additionalProperties"] is False


def test_search_returns_excerpts_and_the_pages_they_show() -> None:
    outcome = run_tool(TOOLS, call("search_policy", query="dental"))

    assert not outcome.result.is_error
    assert outcome.result.call_id == "call_1"
    assert '<excerpt page="58" section="What isn\'t covered">' in outcome.result.content
    assert '<excerpt page="17-18" section="Signing up">' in outcome.result.content
    assert outcome.pages_seen == {58: "What isn't covered", 17: "Signing up", 18: "Signing up"}


def test_search_with_no_results_is_not_an_error() -> None:
    tools = PolicyTools(FakeStore([]), FakePageStore([]))
    outcome = run_tool(tools, call("search_policy", query="passport"))
    assert not outcome.result.is_error
    assert outcome.pages_seen == {}


def test_get_page_returns_the_full_page() -> None:
    outcome = run_tool(TOOLS, call("get_page", page_number=58))

    assert outcome.result.content == (
        '<page number="58" source="handbook.pdf">\nFull text of page 58.\n</page>'
    )
    assert outcome.pages_seen == {58: "Full page"}


def test_get_page_accepts_a_numeric_string() -> None:
    assert not run_tool(TOOLS, call("get_page", page_number="58")).result.is_error


def test_problems_come_back_as_errors_for_the_model() -> None:
    cases = [
        call("get_page", page_number=999),
        call("get_page", page_number="sixty"),
        call("search_policy", query="  "),
        call("search_policy"),
        call("delete_everything"),
    ]
    for bad_call in cases:
        outcome = run_tool(TOOLS, bad_call)
        assert outcome.result.is_error, bad_call
        assert outcome.pages_seen == {}
