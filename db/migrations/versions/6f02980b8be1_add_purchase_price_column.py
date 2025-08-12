"""add_purchase_price_column

Revision ID: YOUR_REVISION_ID
Revises: f1606789faeb
Create Date: 2025-01-XX XX:XX:XX.XXXXXX

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'YOUR_REVISION_ID'  # Keep whatever ID was generated
down_revision = 'f1606789faeb'
branch_labels = None
depends_on = None


def upgrade():
    # Add purchase_price column to holdings table
    op.add_column('holdings', sa.Column('purchase_price', sa.Float(), nullable=True))


def downgrade():
    # Remove purchase_price column from holdings table
    op.drop_column('holdings', 'purchase_price')