"""initial schema

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-10-06 16:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. workspaces
    op.create_table(
        'workspaces',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('slug', sa.String(length=64), nullable=False, unique=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    )

    # 2. agents
    op.create_table(
        'agents',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('workspace_id', sa.String(length=64), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, index=True),
        sa.Column('published_version_id', sa.String(length=64), nullable=True),
        sa.Column('current_version_no', sa.Integer(), nullable=False, default=1),
        sa.Column('draft_voice_id', sa.String(length=64), nullable=False),
        sa.Column('draft_greeting_text', sa.Text(), nullable=False),
        sa.Column('draft_greeting_mode', sa.String(length=32), nullable=False),
        sa.Column('draft_system_prompt', sa.Text(), nullable=False),
        sa.Column('draft_ending_text', sa.Text(), nullable=False),
        sa.Column('draft_end_silence_sec', sa.Integer(), nullable=False, default=20),
        sa.Column('draft_max_duration_sec', sa.Integer(), nullable=False, default=600),
        sa.Column('draft_timezone', sa.String(length=64), nullable=False, default='Asia/Kolkata'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    )

    # 3. agent_versions
    op.create_table(
        'agent_versions',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('agent_id', sa.String(length=64), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('version_no', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('voice_id', sa.String(length=64), nullable=False),
        sa.Column('greeting_text', sa.Text(), nullable=False),
        sa.Column('greeting_mode', sa.String(length=32), nullable=False),
        sa.Column('system_prompt', sa.Text(), nullable=False),
        sa.Column('ending_text', sa.Text(), nullable=False),
        sa.Column('end_silence_sec', sa.Integer(), nullable=False),
        sa.Column('max_duration_sec', sa.Integer(), nullable=False),
        sa.Column('timezone', sa.String(length=64), nullable=False),
        sa.Column('compiled_prompt', sa.Text(), nullable=False),
        sa.Column('compiled_token_count', sa.Integer(), nullable=False),
        sa.Column('change_note', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    )
    op.create_index('idx_agent_version_unique', 'agent_versions', ['agent_id', 'version_no'], unique=True)

    # 4. call_sessions
    op.create_table(
        'call_sessions',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('agent_id', sa.String(length=64), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('agent_version_id', sa.String(length=64), sa.ForeignKey('agent_versions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_sec', sa.Float(), nullable=False, default=0.0),
        sa.Column('end_reason', sa.String(length=64), nullable=False),
        sa.Column('ttfa_ms', sa.Float(), nullable=False, default=0.0),
        sa.Column('handshake_ms', sa.Float(), nullable=False, default=0.0),
    )

    # 5. call_turns
    op.create_table(
        'call_turns',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('session_id', sa.String(length=64), sa.ForeignKey('call_sessions.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('idx', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('started_ms', sa.Float(), nullable=False, default=0.0),
    )


def downgrade() -> None:
    op.drop_table('call_turns')
    op.drop_table('call_sessions')
    op.drop_index('idx_agent_version_unique', table_name='agent_versions')
    op.drop_table('agent_versions')
    op.drop_table('agents')
    op.drop_table('workspaces')
