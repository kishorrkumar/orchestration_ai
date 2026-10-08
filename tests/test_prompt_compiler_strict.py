"""
Unit tests verifying strict prompt compilation, template variable resolution,
sanitization of markdown and scripts, and token budget bounds.
"""

import pytest
from orchestration.prompts.compiler import (
    TemplateResolutionError,
    compile_prompt,
    sanitize_prompt_text,
)


def test_strict_template_resolution_defaults():
    """Verify default variables (company='BrightNet') resolve cleanly with zero stray brackets."""
    template = "You are Alex at {{company}}. Help {{customer_name}} with their router."
    res = compile_prompt(
        system_prompt=template,
        agent_name="Alex",
        variables={"customer_name": "Rohan"},
    )
    assert "{{" not in res.formatted_prompt
    assert "}}" not in res.formatted_prompt
    assert "BrightNet" in res.formatted_prompt
    assert "Rohan" in res.formatted_prompt


def test_strict_template_resolution_missing_variable_raises():
    """Verify missing required variables raise TemplateResolutionError (HTTP 422)."""
    template = "You are helping {{unresolved_custom_field}} at {{company}}."
    with pytest.raises(TemplateResolutionError) as exc_info:
        compile_prompt(
            system_prompt=template,
            agent_name="TestAgent",
            variables={},
            strict=True,
        )
    assert "unresolved_custom_field" in str(exc_info.value)
    assert exc_info.value.missing_variables == ["unresolved_custom_field"]


def test_no_double_brackets_in_output():
    """Fails if any '{{' or '}}' appears in the compiled prompt output."""
    template = (
        "You are {{agent_name}} from {{company}} support. "
        "It is {{day_part}} for {{caller_name}}."
    )
    res = compile_prompt(
        system_prompt=template,
        agent_name="Aarav",
        caller_name="Priya",
    )
    assert "{{" not in res.formatted_prompt
    assert "}}" not in res.formatted_prompt


def test_prompt_sanitization_strips_quotes_and_markdown():
    """Verify markdown bold/headings, emojis, and scripted 'Start:'/'Close:' are removed."""
    messy_prompt = """
    # Customer Support Rules
    * Rule 1: Always listen
    Start: "Hi there, how are you?"
    Close: "Goodbye and have a nice day!"
    You are **patient** and kind 😊.
    """
    clean, removed = sanitize_prompt_text(messy_prompt)
    assert "Start:" not in clean
    assert "Close:" not in clean
    assert "**" not in clean
    assert "#" not in clean
    assert "😊" not in clean
    assert len(removed) > 0


def test_prompt_token_budget_bounds():
    """Verify concise prompts fall within 80-250 token budget."""
    alex_prompt = (
        "You are Alex, a friendly and patient customer support agent at BrightNet, "
        "an internet service provider. You are on a phone call with a customer who is "
        "having trouble with their service. Greet the customer once, warmly, then listen. "
        "Ask what is going wrong, one simple question at a time, and guide them through "
        "one small step at a time, checking that it worked before moving on."
    )
    res = compile_prompt(alex_prompt, agent_name="Alex")
    assert 40 <= res.token_count <= 250
    assert not res.is_critical_overflow
