"""Smoke tests for the Streamlit app with fake stores and models (no index, no network)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import policy_rag.llm.factory
import policy_rag.store.factory
from fakes import FakeLLM, FakePageStore, FakeStore, make_chunk

APP = str(Path(__file__).resolve().parent.parent / "src" / "policy_rag" / "app.py")
PENALTY = make_chunk("c1", "Part B late enrollment penalty", "Premium goes up 10% per year late.", 23, 23)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    st.cache_resource.clear()
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr(policy_rag.store.factory, "create_vector_store", lambda settings: FakeStore([PENALTY]))
    monkeypatch.setattr(policy_rag.store.factory, "create_page_store", lambda settings: FakePageStore([]))
    return AppTest.from_file(APP, default_timeout=10)


def use_fake_model(monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    monkeypatch.setattr(policy_rag.llm.factory, "create_llm_client", lambda settings: FakeLLM(reply))


def markdown(at: AppTest) -> str:
    return "\n".join(element.value for element in at.markdown)


def test_empty_page_offers_example_questions(app: AppTest) -> None:
    app.run()
    assert not app.exception
    assert len(app.button) == 3


def test_rag_answer_shows_the_cited_source(app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_model(monkeypatch, "It goes up 10% for each year late [p. 23].")
    app.run()
    app.radio[0].set_value("rag")
    app.chat_input[0].set_value("What's the Part B penalty?").run()

    assert not app.exception
    page = markdown(app)
    assert "It goes up 10% for each year late [p. 23]." in page
    assert "**p. 23** — Part B late enrollment penalty" in page
    assert len(app.button) == 0  # examples are gone once a question is asked


def test_uncited_answer_shows_a_warning(app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_model(monkeypatch, "It goes up every year.")
    app.run()
    app.radio[0].set_value("rag")
    app.chat_input[0].set_value("What's the Part B penalty?").run()

    assert "cites no pages" in app.warning[0].value


def test_choosing_gemini_without_a_key_shows_an_error(app: AppTest) -> None:
    app.run()
    app.selectbox[0].set_value("gemini")
    app.chat_input[0].set_value("What's the Part B penalty?").run()

    assert not app.exception
    assert "GEMINI_API_KEY" in app.error[0].value
