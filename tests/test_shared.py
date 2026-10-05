"""Unit tests for orchestration.shared foundational modules."""

from orchestration.shared.errors import (
    NotFoundError,
    ProblemDetail,
    PromptTooLongError,
)
from orchestration.shared.ids import (
    new_agent_id,
    new_error_id,
    new_session_id,
    new_turn_id,
)
from orchestration.shared.logging import redact_sensitive_data
from orchestration.shared.settings import Settings


def test_id_generators():
    err_id = new_error_id()
    agt_id = new_agent_id()
    ses_id = new_session_id()
    trn_id = new_turn_id()

    assert err_id.startswith("err_")
    assert agt_id.startswith("agt_")
    assert ses_id.startswith("ses_")
    assert trn_id.startswith("trn_")

    assert len(set([new_agent_id() for _ in range(50)])) == 50


def test_domain_error_problem_details():
    err = NotFoundError("Agent not found")
    problem = err.to_problem()

    assert isinstance(problem, ProblemDetail)
    assert problem.status == 404
    assert problem.code == "NOT_FOUND"
    assert problem.title == "Resource Not Found"
    assert problem.detail == "Agent not found"
    assert problem.error_id.startswith("err_")


def test_validation_error_with_params():
    invalid_params = [{"name": "system_prompt", "reason": "380 tokens > 350 limit"}]
    err = PromptTooLongError("Prompt exceeds maximum length", invalid_params=invalid_params)
    problem = err.to_problem()

    assert problem.status == 422
    assert problem.code == "PROMPT_TOO_LONG"
    assert problem.invalid_params == invalid_params


def test_redaction_logging():
    event_dict = {
        "event": "user_action",
        "api_key": "secret12345",
        "password": "my_password",
        "token": "token_abc",
        "agent_name": "Friendly Caller",
    }
    redacted = redact_sensitive_data(None, "info", event_dict)

    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["password"] == "[REDACTED]"
    assert redacted["token"] == "[REDACTED]"
    assert redacted["agent_name"] == "Friendly Caller"


def test_settings_defaults():
    s = Settings()
    assert s.host == "127.0.0.1"
    assert s.port == 8000
    assert s.client_sample_rate == 16000
    assert s.model_sample_rate == 24000
    assert s.max_prompt_tokens == 350
    assert s.auto_mock_worker is True
