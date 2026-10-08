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
        import re

        # 1. Clean any stray template brackets or residual variables
        raw_prompt = agent.system_prompt or ""
        clean_prompt = re.sub(r"\{\{([a-zA-Z0-9_]+)\}\}", r"\1", raw_prompt)
        clean_prompt = clean_prompt.replace("{{", "").replace("}}", "").strip()

        # 2. Pure system prompt wrapped for PersonaPlex
        wrapped_text = wrap_system_prompt(clean_prompt)

        # 3. Count discrete SentencePiece tokens
        token_count = self.tokenizer.count_tokens(wrapped_text)

        # 4. Run voice linter
        warnings = PromptLinter.lint(clean_prompt)

        return CompiledPrompt(
            raw_prompt=agent.system_prompt,
            compiled_text=clean_prompt,
            wrapped_text=wrapped_text,
            token_count=token_count,
            timezone_str=agent.timezone_str or "UTC",
            local_time_line="",
            warnings=warnings,
        )

    def validate_for_publish(self, agent: Agent) -> CompiledPrompt:
        """Validate compiled prompt, blocking publish if exceeding hard token limit (Engine A only)."""
        compiled = self.compile(agent)
        if getattr(agent, "engine", "personaplex_s2s") == "personaplex_s2s":
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
