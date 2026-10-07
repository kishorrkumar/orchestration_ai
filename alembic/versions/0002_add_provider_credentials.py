"""add provider_credentials table

Revision ID: 0002_add_provider_credentials
Revises: 0001_initial_schema
Create Date: 2026-10-06 16:15:30.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002_add_provider_credentials'
down_revision: Union[str, None] = '0001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'provider_credentials',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('workspace_id', sa.String(length=64), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('provider_id', sa.String(length=64), nullable=False, index=True),
        sa.Column('label', sa.String(length=128), nullable=False, server_default=''),
        sa.Column('ciphertext', sa.Text(), nullable=False),
        sa.Column('nonce', sa.String(length=64), nullable=False),
        sa.Column('key_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('last4', sa.String(length=16), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='untested'),
        sa.Column('last_tested_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_latency_ms', sa.Float(), nullable=True),
        sa.Column('config_json', sa.Text(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    )
    op.create_index(
        'idx_workspace_provider_unique',
        'provider_credentials',
        ['workspace_id', 'provider_id'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index('idx_workspace_provider_unique', table_name='provider_credentials')
    op.drop_table('provider_credentials')
