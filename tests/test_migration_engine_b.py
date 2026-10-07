"""
Tests for reversible Alembic migration: 0003_add_engine_and_turn_metrics.
Tests upgrade, schema verification, downgrade, and re-upgrade.
"""

from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command


@pytest.fixture
def temp_alembic_cfg(tmp_path: Path):
    db_path = tmp_path / "test_migration_0003.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    return cfg, db_path


def test_engine_b_migration_up_down(temp_alembic_cfg):
    cfg, db_path = temp_alembic_cfg

    # 1. Upgrade to head
    command.upgrade(cfg, "head")

    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    inspector = inspect(engine)

    # Verify columns on agents
    agent_cols = {col["name"]: col for col in inspector.get_columns("agents")}
    assert "draft_engine" in agent_cols
    assert "draft_language" in agent_cols
    assert "draft_pipeline_json" in agent_cols

    # Verify columns on agent_versions
    ver_cols = {col["name"]: col for col in inspector.get_columns("agent_versions")}
    assert "engine" in ver_cols
    assert "language" in ver_cols
    assert "pipeline_json" in ver_cols

    # Verify columns on call_sessions
    ses_cols = {col["name"]: col for col in inspector.get_columns("call_sessions")}
    assert "engine" in ses_cols
    assert "providers_used_json" in ses_cols

    # Verify columns on call_turns
    turn_cols = {col["name"]: col for col in inspector.get_columns("call_turns")}
    assert "eot_ms" in turn_cols
    assert "stt_ms" in turn_cols
    assert "llm_ttft_ms" in turn_cols
    assert "tts_ttfa_ms" in turn_cols
    assert "voice_to_voice_ms" in turn_cols

    # 2. Downgrade to 0002_add_provider_credentials
    command.downgrade(cfg, "0002_add_provider_credentials")

    inspector_down = inspect(engine)
    agent_cols_down = {col["name"]: col for col in inspector_down.get_columns("agents")}
    assert "draft_engine" not in agent_cols_down
    assert "draft_pipeline_json" not in agent_cols_down

    ver_cols_down = {col["name"]: col for col in inspector_down.get_columns("agent_versions")}
    assert "engine" not in ver_cols_down
    assert "pipeline_json" not in ver_cols_down

    ses_cols_down = {col["name"]: col for col in inspector_down.get_columns("call_sessions")}
    assert "engine" not in ses_cols_down

    turn_cols_down = {col["name"]: col for col in inspector_down.get_columns("call_turns")}
    assert "llm_ttft_ms" not in turn_cols_down

    # 3. Re-upgrade to head
    command.upgrade(cfg, "head")
    inspector_re = inspect(engine)
    agent_cols_re = {col["name"]: col for col in inspector_re.get_columns("agents")}
    assert "draft_engine" in agent_cols_re
