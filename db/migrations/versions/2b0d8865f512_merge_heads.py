"""merge heads

Revision ID: 2b0d8865f512
Revises: add_agent_debates, fix_portfolio_risk_snapshots
Create Date: 2025-08-27 10:11:25.813828

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2b0d8865f512'
down_revision: Union[str, Sequence[str], None] = ('add_agent_debates', 'fix_portfolio_risk_snapshots')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
