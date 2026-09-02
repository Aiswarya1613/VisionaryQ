from types import SimpleNamespace

import pytest

import app.services.llm_service as llm_module
from app.services.llm_service import LLMService


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

def test_llm_service_uses_central_config(
    monkeypatch
):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            ollama_model="test-config-model"
        ),
    )

    service = LLMService()

    assert service.model == (
        "test-config-model"
    )


def test_llm_service_allows_model_override(
    monkeypatch
):
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            ollama_model="config-model"
        ),
    )

    service = LLMService(
        model="override-model"
    )

    assert service.model == (
        "override-model"
    )


# ---------------------------------------------------------
# Validation / guardrails
# ---------------------------------------------------------

def test_generate_rejects_empty_query():
    service = LLMService(
        model="test-model"
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty"
    ):
        service.generate(
            query="   ",
            context="Some context."
        )


def test_generate_returns_fallback_for_empty_context(
    monkeypatch
):
    def fail_if_called(*args, **kwargs):
        pytest.fail(
            "Ollama should not be called "
            "when context is empty."
        )

    monkeypatch.setattr(
        llm_module.ollama,
        "generate",
        fail_if_called,
    )

    service = LLMService(
        model="test-model"
    )

    result = service.generate(
        query="What is generative AI?",
        context="   ",
    )

    assert result == (
        "I could not find relevant information "
        "in the available video content."
    )


# ---------------------------------------------------------
# Ollama generation
# ---------------------------------------------------------

def test_generate_calls_ollama_with_grounded_prompt(
    monkeypatch
):
    captured = {}

    def fake_generate(
        model,
        prompt
    ):
        captured["model"] = model
        captured["prompt"] = prompt

        return {
            "response": (
                "  Generative AI creates "
                "new content.  "
            )
        }

    monkeypatch.setattr(
        llm_module.ollama,
        "generate",
        fake_generate,
    )

    service = LLMService(
        model="test-model"
    )

    result = service.generate(
        query="What is generative AI?",
        context=(
            "Generative AI creates "
            "new content."
        ),
    )

    assert captured["model"] == (
        "test-model"
    )

    assert (
        "Generative AI creates new content."
        in captured["prompt"]
    )

    assert (
        "What is generative AI?"
        in captured["prompt"]
    )

    assert (
        "using ONLY the provided context"
        in captured["prompt"]
    )

    assert (
        "Do not use outside knowledge."
        in captured["prompt"]
    )

    assert result == (
        "Generative AI creates new content."
    )