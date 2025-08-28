"""Add risk detection tables

Revision ID: add_risk_detection_tables  
Revises: 4ee5cfef7485
Create Date: 2025-08-26 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'add_risk_detection_tables'
down_revision = '4ee5cfef7485'  # Your latest revision
branch_labels = None
depends_on = None

def upgrade() -> None:
    """Add risk detection columns and tables"""
    
    # First, add new columns to existing portfolio_risk_snapshots table
    op.add_column('portfolio_risk_snapshots', 
                  sa.Column('compressed_metrics', sa.LargeBinary(), nullable=True))
    op.add_column('portfolio_risk_snapshots', 
                  sa.Column('compression_ratio', sa.Float(), nullable=True))
    op.add_column('portfolio_risk_snapshots', 
                  sa.Column('metrics_summary', postgresql.JSON(), nullable=True))
    op.add_column('portfolio_risk_snapshots', 
                  sa.Column('data_quality_score', sa.Float(), server_default='1.0', nullable=True))
    op.add_column('portfolio_risk_snapshots', 
                  sa.Column('calculation_time_ms', sa.Float(), nullable=True))
    
    # Create risk_trends table
    op.create_table('risk_trends',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portfolio_id', sa.Integer(), nullable=False),
        sa.Column('metric_name', sa.String(length=50), nullable=False),
        sa.Column('current_value', sa.Float(), nullable=False),
        sa.Column('trend_direction', sa.String(length=20), nullable=False),
        sa.Column('trend_strength', sa.Float(), nullable=False),
        sa.Column('forecast_1d', sa.Float(), nullable=True),
        sa.Column('forecast_7d', sa.Float(), nullable=True),
        sa.Column('forecast_30d', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('r_squared', sa.Float(), nullable=True),
        sa.Column('p_value', sa.Float(), nullable=True),
        sa.Column('analysis_period_days', sa.Integer(), server_default='90', nullable=True),
        sa.Column('data_points_used', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create risk_thresholds table  
    op.create_table('risk_thresholds',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portfolio_id', sa.Integer(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('metric_name', sa.String(length=50), nullable=False),
        sa.Column('warning_threshold', sa.Float(), nullable=False),
        sa.Column('critical_threshold', sa.Float(), nullable=False),
        sa.Column('emergency_threshold', sa.Float(), nullable=True),
        sa.Column('direction', sa.String(length=20), server_default="'increase'", nullable=False),
        sa.Column('enabled', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('lookback_periods', sa.Integer(), server_default='5', nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['portfolio_id'], ['portfolios.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create price_data_cache table
    op.create_table('price_data_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('currency', sa.String(length=10), server_default="'USD'", nullable=True),
        sa.Column('volume', sa.Float(), nullable=True),
        sa.Column('change', sa.Float(), nullable=True),
        sa.Column('change_percent', sa.Float(), nullable=True),
        sa.Column('provider', sa.String(length=30), nullable=False),
        sa.Column('data_quality_score', sa.Float(), server_default='1.0', nullable=True),
        sa.Column('market_timestamp', sa.DateTime(), nullable=True),
        sa.Column('fetched_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('bid_price', sa.Float(), nullable=True),
        sa.Column('ask_price', sa.Float(), nullable=True),
        sa.Column('market_cap', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_risk_trends_portfolio_metric', 'risk_trends', ['portfolio_id', 'metric_name'])
    op.create_index('idx_price_data_cache_symbol_expires', 'price_data_cache', ['symbol', 'expires_at'])
    
    # Insert default thresholds
    op.execute("""
        INSERT INTO risk_thresholds 
        (portfolio_id, user_id, metric_name, warning_threshold, critical_threshold, emergency_threshold, direction) 
        VALUES 
        (NULL, NULL, 'annualized_volatility', 0.25, 0.35, 0.50, 'increase'),
        (NULL, NULL, 'var_95', -0.05, -0.08, -0.12, 'decrease'),
        (NULL, NULL, 'sharpe_ratio', 0.8, 0.5, 0.2, 'decrease')
    """)

def downgrade() -> None:
    """Remove risk detection tables and columns"""
    op.drop_index('idx_price_data_cache_symbol_expires', table_name='price_data_cache')
    op.drop_index('idx_risk_trends_portfolio_metric', table_name='risk_trends')
    op.drop_table('price_data_cache')
    op.drop_table('risk_thresholds')
    op.drop_table('risk_trends')
    
    # Remove added columns
    op.drop_column('portfolio_risk_snapshots', 'calculation_time_ms')
    op.drop_column('portfolio_risk_snapshots', 'data_quality_score')
    op.drop_column('portfolio_risk_snapshots', 'metrics_summary')
    op.drop_column('portfolio_risk_snapshots', 'compression_ratio')
    op.drop_column('portfolio_risk_snapshots', 'compressed_metrics')