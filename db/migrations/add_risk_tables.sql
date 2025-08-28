-- db/migrations/add_risk_tables.sql
-- Risk Detection Pipeline Database Migration
-- Task 2.1: Add tables for risk snapshot storage and analysis

-- Table 1: Portfolio Risk Snapshots (compressed storage)
CREATE TABLE portfolio_risk_snapshots (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    snapshot_date TIMESTAMP NOT NULL,
    compressed_metrics BYTEA NOT NULL,
    compression_ratio REAL,
    metrics_summary JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    data_quality_score REAL DEFAULT 1.0,
    calculation_time_ms REAL,
    
    CONSTRAINT fk_portfolio_risk_snapshots_portfolio 
        FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE
);

-- Table 2: Risk Trends (trend analysis and forecasting)
CREATE TABLE risk_trends (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    current_value REAL NOT NULL,
    trend_direction VARCHAR(20) NOT NULL,
    trend_strength REAL NOT NULL,
    forecast_1d REAL,
    forecast_7d REAL,
    forecast_30d REAL,
    confidence_score REAL NOT NULL,
    r_squared REAL,
    p_value REAL,
    analysis_period_days INTEGER DEFAULT 90,
    data_points_used INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_risk_trends_portfolio 
        FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE
);

-- Table 3: Risk Change Events (alerts and significant changes)
CREATE TABLE risk_change_events (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    user_id INTEGER,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    current_value REAL NOT NULL,
    previous_value REAL,
    threshold_breached REAL,
    change_magnitude REAL NOT NULL,
    statistical_significance REAL,
    confidence_score REAL NOT NULL,
    detection_method VARCHAR(30) NOT NULL,
    message TEXT NOT NULL,
    recommendations JSONB,
    workflow_triggered BOOLEAN DEFAULT FALSE,
    workflow_response TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    acknowledged_at TIMESTAMP,
    acknowledged_by INTEGER,
    resolved_at TIMESTAMP,
    
    CONSTRAINT fk_risk_change_events_portfolio 
        FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    CONSTRAINT fk_risk_change_events_user 
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT fk_risk_change_events_acknowledged_by 
        FOREIGN KEY (acknowledged_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Table 4: Risk Thresholds (configurable thresholds)
CREATE TABLE risk_thresholds (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER,  -- NULL = global default
    user_id INTEGER,
    metric_name VARCHAR(50) NOT NULL,
    warning_threshold REAL NOT NULL,
    critical_threshold REAL NOT NULL,
    emergency_threshold REAL,
    direction VARCHAR(20) DEFAULT 'increase' NOT NULL,
    enabled BOOLEAN DEFAULT TRUE NOT NULL,
    lookback_periods INTEGER DEFAULT 5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    
    CONSTRAINT fk_risk_thresholds_portfolio 
        FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
    CONSTRAINT fk_risk_thresholds_user 
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_risk_thresholds_created_by 
        FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Table 5: Price Data Cache (performance optimization)
CREATE TABLE price_data_cache (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    price REAL NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    volume REAL,
    change_value REAL,
    change_percent REAL,
    provider VARCHAR(30) NOT NULL,
    data_quality_score REAL DEFAULT 1.0,
    market_timestamp TIMESTAMP,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    bid_price REAL,
    ask_price REAL,
    market_cap REAL
);

-- Performance Indexes
CREATE INDEX idx_portfolio_risk_snapshots_portfolio_date 
    ON portfolio_risk_snapshots(portfolio_id, snapshot_date);

CREATE INDEX idx_portfolio_risk_snapshots_created_at 
    ON portfolio_risk_snapshots(created_at);

CREATE INDEX idx_risk_trends_portfolio_metric 
    ON risk_trends(portfolio_id, metric_name);

CREATE INDEX idx_risk_trends_last_updated 
    ON risk_trends(last_updated);

CREATE INDEX idx_risk_change_events_portfolio_severity 
    ON risk_change_events(portfolio_id, severity, detected_at);

CREATE INDEX idx_risk_change_events_detected_at 
    ON risk_change_events(detected_at);

CREATE INDEX idx_risk_change_events_user_severity 
    ON risk_change_events(user_id, severity, detected_at);

CREATE INDEX idx_risk_change_events_unacknowledged 
    ON risk_change_events(acknowledged_at) WHERE acknowledged_at IS NULL;

CREATE INDEX idx_risk_thresholds_portfolio_metric 
    ON risk_thresholds(portfolio_id, metric_name) WHERE enabled = TRUE;

CREATE INDEX idx_risk_thresholds_user_metric 
    ON risk_thresholds(user_id, metric_name) WHERE enabled = TRUE;

CREATE INDEX idx_price_data_cache_symbol_expires 
    ON price_data_cache(symbol, expires_at);

CREATE INDEX idx_price_data_cache_expires_at 
    ON price_data_cache(expires_at);

-- Insert default risk thresholds
INSERT INTO risk_thresholds (
    portfolio_id, user_id, metric_name, warning_threshold, critical_threshold, 
    emergency_threshold, direction, created_at
) VALUES 
-- Volatility thresholds
(NULL, NULL, 'annualized_volatility', 0.25, 0.35, 0.50, 'increase', CURRENT_TIMESTAMP),
(NULL, NULL, 'annualized_volatility', 0.05, 0.03, 0.01, 'decrease', CURRENT_TIMESTAMP),

-- VaR thresholds  
(NULL, NULL, 'var_95', -0.05, -0.08, -0.12, 'decrease', CURRENT_TIMESTAMP),
(NULL, NULL, 'cvar_95', -0.08, -0.12, -0.18, 'decrease', CURRENT_TIMESTAMP),

-- Drawdown thresholds
(NULL, NULL, 'max_drawdown', -0.15, -0.25, -0.35, 'decrease', CURRENT_TIMESTAMP),

-- Risk-adjusted return thresholds
(NULL, NULL, 'sharpe_ratio', 0.8, 0.5, 0.2, 'decrease', CURRENT_TIMESTAMP),
(NULL, NULL, 'sortino_ratio', 1.0, 0.6, 0.3, 'decrease', CURRENT_TIMESTAMP),

-- Distribution risk thresholds
(NULL, NULL, 'skewness', -1.0, -1.5, -2.0, 'decrease', CURRENT_TIMESTAMP),
(NULL, NULL, 'kurtosis', 3.0, 5.0, 8.0, 'increase', CURRENT_TIMESTAMP);

-- Add triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_risk_thresholds_updated_at 
    BEFORE UPDATE ON risk_thresholds 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_risk_trends_updated_at 
    BEFORE UPDATE ON risk_trends 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE portfolio_risk_snapshots IS 'Compressed storage for portfolio risk snapshots with historical data';
COMMENT ON TABLE risk_trends IS 'Risk trend analysis and forecasting data';
COMMENT ON TABLE risk_change_events IS 'Significant risk change events and alerts';
COMMENT ON TABLE risk_thresholds IS 'Configurable risk thresholds for different metrics and portfolios';
COMMENT ON TABLE price_data_cache IS 'Cached price data for performance optimization';

COMMENT ON COLUMN portfolio_risk_snapshots.compressed_metrics IS 'Gzip-compressed JSON of complete risk metrics';
COMMENT ON COLUMN portfolio_risk_snapshots.metrics_summary IS 'Quick-access summary of key metrics (not compressed)';
COMMENT ON COLUMN risk_change_events.workflow_triggered IS 'Whether this event triggered an AI workflow';
COMMENT ON COLUMN risk_thresholds.direction IS 'Threshold direction: increase, decrease, or absolute';
COMMENT ON COLUMN price_data_cache.expires_at IS 'When this cached price data expires';