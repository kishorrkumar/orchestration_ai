"""add engine, language, pipeline and turn metrics

Revision ID: 0003_add_engine_and_turn_metrics
Revises: 0002_add_provider_credentials
Create Date: 2026-10-06 16:45:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0003_add_engine_and_turn_metrics'
down_revision: Union[str, None] = '0002_add_provider_credentials'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('agents') as batch_op:
        batch_op.add_column(sa.Column('draft_engine', sa.String(length=32), nullable=False, server_default='personaplex_s2s'))
        batch_op.add_column(sa.Column('draft_language', sa.String(length=32), nullable=False, server_default='en'))
        batch_op.add_column(sa.Column('draft_pipeline_json', sa.Text(), nullable=False, server_default='{}'))

    with op.batch_alter_table('agent_versions') as batch_op:
        batch_op.add_column(sa.Column('engine', sa.String(length=32), nullable=False, server_default='personaplex_s2s'))
        batch_op.add_column(sa.Column('language', sa.String(length=32), nullable=False, server_default='en'))
        batch_op.add_column(sa.Column('pipeline_json', sa.Text(), nullable=False, server_default='{}'))

    with op.batch_alter_table('call_sessions') as batch_op:
        batch_op.add_column(sa.Column('engine', sa.String(length=32), nullable=False, server_default='personaplex_s2s'))
        batch_op.add_column(sa.Column('providers_used_json', sa.Text(), nullable=False, server_default='{}'))

    with op.batch_alter_table('call_turns') as batch_op:
        batch_op.add_column(sa.Column('eot_ms', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('stt_ms', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('llm_ttft_ms', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('tts_ttfa_ms', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('voice_to_voice_ms', sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('call_turns') as batch_op:
        batch_op.drop_column('voice_to_voice_ms')
        batch_op.drop_column('tts_ttfa_ms')
        batch_op.drop_column('llm_ttft_ms')
        batch_op.drop_column('stt_ms')
        batch_op.drop_column('eot_ms')

    with op.batch_alter_table('call_sessions') as batch_op:
        batch_op.drop_column('providers_used_json')
        batch_op.drop_column('engine')

    with op.batch_alter_table('agent_versions') as batch_op:
        batch_op.drop_column('pipeline_json')
        batch_op.drop_column('language')
        batch_op.drop_column('engine')

    with op.batch_alter_table('agents') as batch_op:
        batch_op.drop_column('draft_pipeline_json')
        batch_op.drop_column('draft_language')
        batch_op.drop_column('draft_engine')
