"""
Application use case for compiling and linting voice prompts.
Enforces ordering:
1. Core system prompt instructions
2. Injected local date/time line
3. "Start: <greeting>"
4. "Close: <ending>"
Followed by SentencePiece token counting and voice linting.
"""

from __future__ import annotations

from orchestration.domain.agent import Agent
from orchestration.domain.prompt import (
    HARD_TOKEN_LIMIT,
    CompiledPrompt,
    PromptLinter,
    compute_local_time_context,
    wrap_system_prompt,
)
from orchestration.domain.protocols import Clock, Tokenizer
from orchestration.shared.errors import PromptTooLongError


class PromptCompilerUseCase:
    """Use case coordinating prompt compilation, token budgeting, and linting."""

    def __init__(self, tokenizer: Tokenizer, clock: Clock) -> None:
        self.tokenizer = tokenizer
        self.clock = clock

    def compile(
        self,
        agent: Agent,
        customer_name: str = "Friend",
        company: str = "PersonaPlex",
    ) -> CompiledPrompt:
        """Compile an Agent's prompt fields into the exact string consumed by PersonaPlex."""
        # 1. Compute time context
        now = self.clock.now_utc()
        local_time_line, day_part = compute_local_time_context(agent.timezone_str, now)

        # 2. Template variable substitution
        replacements = {
            "{{agent_name}}": agent.name,
            "{{customer_name}}": customer_name,
            "{{company}}": company,
            "{{day_part}}": day_part,
            "{{timezone}}": agent.timezone_str,
            "{{time}}": local_time_line,
        }

        rendered_prompt = agent.system_prompt
        rendered_greeting = agent.greeting
        rendered_ending = agent.ending

        for var, val in replacements.items():
            rendered_prompt = rendered_prompt.replace(var, val)
            rendered_greeting = rendered_greeting.replace(var, val)
            rendered_ending = rendered_ending.replace(var, val)

        # 3. Assemble structured body
        lines = [
            rendered_prompt.strip(),
            local_time_line,
        ]
        if rendered_greeting.strip():
            lines.append(f"Start: {rendered_greeting.strip()}")
        if rendered_ending.strip():
            lines.append(f"Close: {rendered_ending.strip()}")

        raw_compiled = "\n\n".join(lines)
        wrapped_text = wrap_system_prompt(raw_compiled)

        # 4. Count discrete SentencePiece tokens
        token_count = self.tokenizer.count_tokens(wrapped_text)

        # 5. Run voice linter
        warnings = PromptLinter.lint(raw_compiled)

        return CompiledPrompt(
            raw_prompt=agent.system_prompt,
            compiled_text=raw_compiled,
            wrapped_text=wrapped_text,
            token_count=token_count,
            timezone_str=agent.timezone_str,
            local_time_line=local_time_line,
            warnings=warnings,
        )

    def validate_for_publish(self, agent: Agent) -> CompiledPrompt:
        """Validate compiled prompt, blocking publish if exceeding hard token limit."""
        compiled = self.compile(agent)
        if not compiled.is_within_hard_limit:
            raise PromptTooLongError(
                f"Compiled prompt has {compiled.token_count} tokens, exceeding hard limit of {HARD_TOKEN_LIMIT}.",
                invalid_params=[
                    {
                        "name": "system_prompt",
                        "token_count": compiled.token_count,
                        "limit": HARD_TOKEN_LIMIT,
                    }
                ],
            )
        return compiled
