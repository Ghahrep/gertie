"""Fix portfolio risk snapshots missing columns

Revision ID: fix_portfolio_risk_snapshots
Revises: add_risk_detection_tables
Create Date: 2025-08-26 16:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = 'fix_portfolio_risk_snapshots'
down_revision = 'add_risk_detection_tables'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Add missing columns that the model expects
    op.add_column('portfolio_risk_snapshots', 
                  sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False))
    
def downgrade() -> None:
    op.drop_column('portfolio_risk_snapshots', 'created_at')