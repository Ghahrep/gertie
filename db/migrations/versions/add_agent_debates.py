"""Add agent debate tables

Create this file as: db/migrations/versions/add_agent_debates.py

Revision ID: add_agent_debates
Revises: a35b9affdfe8
Create Date: 2024-XX-XX XX:XX:XX.XXXXXX

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'add_agent_debates'
down_revision: Union[str, Sequence[str], None] = 'a35b9affdfe8'  # Your latest revision
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Create agent debate tables."""
    
    # Create agent_debates table
    op.create_table('agent_debates',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('debate_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('topic', sa.Text(), nullable=False),
        sa.Column('participants', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='initializing'),
        sa.Column('current_stage', sa.String(50), nullable=True),
        sa.Column('urgency_level', sa.String(10), nullable=False, default='medium'),
        sa.Column('results', sa.JSON(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index(op.f('ix_agent_debates_id'), 'agent_debates', ['id'], unique=False)
    op.create_index(op.f('ix_agent_debates_debate_id'), 'agent_debates', ['debate_id'], unique=True)
    op.create_index(op.f('ix_agent_debates_user_id'), 'agent_debates', ['user_id'], unique=False)
    op.create_index(op.f('ix_agent_debates_status'), 'agent_debates', ['status'], unique=False)
    
    # Create debate_rounds table (optional - for detailed tracking)
    op.create_table('debate_rounds',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('debate_id', sa.String(36), nullable=False),
        sa.Column('round_number', sa.Integer(), nullable=False),
        sa.Column('stage', sa.String(50), nullable=False),
        sa.Column('positions', sa.JSON(), nullable=True),
        sa.Column('challenges', sa.JSON(), nullable=True),
        sa.Column('responses', sa.JSON(), nullable=True),
        sa.Column('round_summary', sa.Text(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['debate_id'], ['agent_debates.debate_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for rounds
    op.create_index(op.f('ix_debate_rounds_id'), 'debate_rounds', ['id'], unique=False)
    op.create_index(op.f('ix_debate_rounds_debate_id'), 'debate_rounds', ['debate_id'], unique=False)

def downgrade() -> None:
    """Drop agent debate tables."""
    op.drop_index(op.f('ix_debate_rounds_debate_id'), table_name='debate_rounds')
    op.drop_index(op.f('ix_debate_rounds_id'), table_name='debate_rounds')
    op.drop_table('debate_rounds')
    
    op.drop_index(op.f('ix_agent_debates_status'), table_name='agent_debates')
    op.drop_index(op.f('ix_agent_debates_user_id'), table_name='agent_debates')
    op.drop_index(op.f('ix_agent_debates_debate_id'), table_name='agent_debates')
    op.drop_index(op.f('ix_agent_debates_id'), table_name='agent_debates')
    op.drop_table('agent_debates')