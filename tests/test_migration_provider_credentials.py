"""
Tests for reversible Alembic migration: 0002_add_provider_credentials.
Tests upgrade, schema verification, downgrade, and re-upgrade.
"""

from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command


@pytest.fixture
def temp_alembic_cfg(tmp_path: Path):
    db_path = tmp_path / "test_migration.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    return cfg, db_path


def test_provider_credentials_migration_up_down(temp_alembic_cfg):
    cfg, db_path = temp_alembic_cfg

    # 1. Upgrade to head
    command.upgrade(cfg, "head")

    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "provider_credentials" in tables
    assert "workspaces" in tables
    assert "agents" in tables

    # Verify columns on provider_credentials
    columns = {col["name"]: col for col in inspector.get_columns("provider_credentials")}
    expected_cols = [
        "id", "workspace_id", "provider_id", "label",
        "ciphertext", "nonce", "key_version", "last4",
        "status", "last_tested_at", "last_latency_ms",
        "config_json", "created_at"
    ]
    for col_name in expected_cols:
        assert col_name in columns, f"Missing column {col_name}"

    # Ensure no plaintext column exists
    assert "api_key" not in columns
    assert "plaintext" not in columns
    assert "key" not in columns

    # 2. Downgrade to 0001_initial_schema
    command.downgrade(cfg, "0001_initial_schema")

    inspector_down = inspect(engine)
    tables_down = inspector_down.get_table_names()
    assert "provider_credentials" not in tables_down
    assert "agents" in tables_down

    # 3. Re-upgrade to head
    command.upgrade(cfg, "head")
    inspector_re = inspect(engine)
    assert "provider_credentials" in inspector_re.get_table_names()
