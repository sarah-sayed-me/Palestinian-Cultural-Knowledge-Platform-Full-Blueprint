import pytest
from fastapi import HTTPException

from src.api import main as api_main
from src.api.main import AskRequest, ask, health
from src.rag.answer import Answer, Citation
from src.rag.config import RagConfig


class _FakePipeline:
    def __init__(self, answer=None, error=None):
        self._answer = answer
        self._error = error
        self.config = RagConfig.default()
        self.last_question = None

    def ask(self, question):
        self.last_question = question
        if self._error:
            raise self._error
        return self._answer


@pytest.fixture(autouse=True)
def _clear_state():
    api_main._state.clear()
    yield
    api_main._state.clear()


def test_health_reports_not_ready_before_startup():
    result = health()

    assert result == {"status": "starting", "pipeline_ready": False}


def test_health_reports_ready_once_pipeline_is_set():
    api_main._state["pipeline"] = object()

    result = health()

    assert result == {"status": "ok", "pipeline_ready": True}


def test_ask_returns_text_and_citations():
    answer = Answer(
        text="الكنافة حلوى فلسطينية شهيرة. [1]",
        citations=[Citation(index=1, doc_id="doc-1", title="كنافة", source_url="https://example.com")],
    )
    pipeline = _FakePipeline(answer=answer)
    api_main._state["pipeline"] = pipeline
    api_main._state["config"] = pipeline.config

    response = ask(AskRequest(question="ما هي الكنافة؟"))

    assert response.text == answer.text
    assert len(response.citations) == 1
    assert response.citations[0].doc_id == "doc-1"
    assert pipeline.last_question == "ما هي الكنافة؟"


def test_ask_rejects_empty_question():
    api_main._state["pipeline"] = _FakePipeline(answer=Answer(text="", citations=[]))
    api_main._state["config"] = RagConfig.default()

    with pytest.raises(HTTPException) as exc_info:
        ask(AskRequest(question="   "))

    assert exc_info.value.status_code == 400


def test_ask_converts_pipeline_runtime_error_to_503():
    pipeline = _FakePipeline(error=RuntimeError("Ollama not reachable"))
    api_main._state["pipeline"] = pipeline
    api_main._state["config"] = pipeline.config

    with pytest.raises(HTTPException) as exc_info:
        ask(AskRequest(question="سؤال"))

    assert exc_info.value.status_code == 503
    assert "Ollama not reachable" in exc_info.value.detail


def test_ask_applies_top_k_override_then_restores_config():
    answer = Answer(text="ok", citations=[])
    pipeline = _FakePipeline(answer=answer)
    original_config = pipeline.config
    api_main._state["pipeline"] = pipeline
    api_main._state["config"] = original_config

    ask(AskRequest(question="سؤال", top_k=3))

    # config is restored to the original object after the request completes
    assert pipeline.config is original_config
