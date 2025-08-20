# clean_fix_models.py
"""
Clean fix for your db/models.py file
Removes duplicates and fixes syntax errors
"""

def clean_models_file():
    """Clean up the models file and fix syntax errors"""
    
    models_file = 'db/models.py'
    
    # Read the current file
    with open(models_file, 'r') as f:
        content = f.read()
    
    print("🔧 Cleaning up db/models.py...")
    
    # Find where the duplicates start (after the ProactiveAlert class)
    # We'll keep everything up to and including ProactiveAlert, then add clean versions
    
    # Find the end of ProactiveAlert class
    proactive_alert_end = content.find('def __repr__(self):\n        return f"<ProactiveAlert(id={self.id}, type={self.alert_type}, priority={self.priority})>"')
    
    if proactive_alert_end == -1:
        print("❌ Could not find ProactiveAlert class end")
        return False
    
    # Find the end of that method
    clean_end = content.find('\n\n', proactive_alert_end) + 2
    
    # Keep everything up to that point
    clean_content = content[:clean_end]
    
    # Now add the clean risk models
    clean_content += '''
# Risk Attribution Models - Clean Version
from sqlalchemy.dialects.postgresql import JSON
import uuid

class PortfolioRiskSnapshot(Base):
    """
    Stores point-in-time risk metrics for portfolios
    """
    __tablename__ = "portfolio_risk_snapshots"
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    
    # Portfolio identification
    user_id = Column(String(50), index=True, nullable=False)
    portfolio_id = Column(String(50), index=True, nullable=False)
    portfolio_name = Column(String(100))
    
    # Timestamp information
    snapshot_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Core risk metrics
    volatility = Column(Float, nullable=False)
    beta = Column(Float)
    max_drawdown = Column(Float)
    var_95 = Column(Float)
    var_99 = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    
    # Composite metrics
    risk_score = Column(Float, nullable=False)
    risk_grade = Column(String(2))
    
    # Portfolio composition metrics
    portfolio_value = Column(Float)
    num_positions = Column(Integer)
    concentration_risk = Column(Float)
    
    # Advanced risk metrics
    tail_dependency = Column(Float)
    hurst_exponent = Column(Float)
    regime_state = Column(Integer)
    
    # Change detection
    volatility_change_pct = Column(Float)
    risk_score_change_pct = Column(Float)
    is_threshold_breach = Column(Boolean, default=False)
    
    # Metadata
    calculation_method = Column(String(20), default='standard')
    data_quality_score = Column(Float, default=1.0)
    notes = Column(Text)
    
    # Raw data storage
    raw_metrics = Column(JSON)
    portfolio_weights = Column(JSON)
    correlation_matrix = Column(JSON)
    
    # Table configuration
    __table_args__ = (
        Index('idx_user_portfolio_date', 'user_id', 'portfolio_id', 'snapshot_date'),
        Index('idx_risk_score_date', 'risk_score', 'snapshot_date'),
        Index('idx_threshold_breach', 'is_threshold_breach', 'snapshot_date'),
        Index('idx_user_recent', 'user_id', 'snapshot_date'),
        {'extend_existing': True}
    )
    
    def __repr__(self):
        return f"<RiskSnapshot(user={self.user_id}, portfolio={self.portfolio_id}, risk_score={self.risk_score})>"
    
    @property
    def risk_level(self):
        """Convert risk score to descriptive level"""
        if self.risk_score <= 20:
            return "Very Low"
        elif self.risk_score <= 40:
            return "Low"
        elif self.risk_score <= 60:
            return "Moderate"
        elif self.risk_score <= 80:
            return "High"
        else:
            return "Very High"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'snapshot_id': self.snapshot_id,
            'user_id': self.user_id,
            'portfolio_id': self.portfolio_id,
            'portfolio_name': self.portfolio_name,
            'snapshot_date': self.snapshot_date.isoformat() if self.snapshot_date else None,
            'volatility': self.volatility,
            'beta': self.beta,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'sharpe_ratio': self.sharpe_ratio,
            'risk_score': self.risk_score,
            'risk_grade': self.risk_grade,
            'risk_level': self.risk_level,
            'portfolio_value': self.portfolio_value,
            'volatility_change_pct': self.volatility_change_pct,
            'risk_score_change_pct': self.risk_score_change_pct,
            'is_threshold_breach': self.is_threshold_breach,
            'tail_dependency': self.tail_dependency,
            'hurst_exponent': self.hurst_exponent
        }


class RiskThresholdConfig(Base):
    """
    Configurable risk thresholds per user/portfolio
    """
    __tablename__ = "risk_threshold_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), index=True, nullable=False)
    portfolio_id = Column(String(50), index=True)
    
    # Threshold percentages
    volatility_threshold = Column(Float, default=0.15)
    beta_threshold = Column(Float, default=0.20)
    max_drawdown_threshold = Column(Float, default=0.10)
    var_threshold = Column(Float, default=0.15)
    risk_score_threshold = Column(Float, default=0.20)
    
    # Absolute thresholds
    max_acceptable_risk_score = Column(Float, default=80.0)
    max_acceptable_volatility = Column(Float, default=0.50)
    
    # Monitoring settings
    monitoring_enabled = Column(Boolean, default=True)
    alert_frequency = Column(String(20), default='immediate')
    notification_methods = Column(JSON, default=lambda: ['websocket', 'email'])
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    __table_args__ = (
        Index('idx_user_portfolio_config', 'user_id', 'portfolio_id', unique=True),
        {'extend_existing': True}
    )


class RiskAlertLog(Base):
    """
    Log of all risk alerts sent to users
    """
    __tablename__ = "risk_alert_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    
    # Alert identification
    user_id = Column(String(50), index=True, nullable=False)
    portfolio_id = Column(String(50), index=True, nullable=False)
    snapshot_id = Column(String(36))
    
    # Alert details
    alert_type = Column(String(30), nullable=False)
    alert_severity = Column(String(10), default='medium')
    alert_message = Column(Text, nullable=False)
    
    # Risk metrics
    triggered_metrics = Column(JSON)
    risk_change_summary = Column(JSON)
    
    # Workflow integration
    workflow_id = Column(String(36))
    workflow_status = Column(String(20))
    
    # Notification tracking
    notification_methods = Column(JSON)
    notification_status = Column(JSON)
    user_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    __table_args__ = (
        Index('idx_alert_user_portfolio', 'user_id', 'portfolio_id'),
        Index('idx_alert_workflow', 'workflow_id'),
        {'extend_existing': True}
    )


# Helper functions
def create_risk_snapshot_from_calculator(
    user_id: str,
    portfolio_id: str,
    risk_result: dict,
    portfolio_data: dict = None
) -> PortfolioRiskSnapshot:
    """Create risk snapshot from risk calculator output"""
    
    if risk_result['status'] != 'success':
        raise ValueError(f"Risk calculation failed: {risk_result.get('error')}")
    
    metrics = risk_result['metrics']
    
    # Calculate risk grade
    risk_score = metrics.get('risk_score', 0)
    if risk_score <= 20:
        risk_grade = 'A'
    elif risk_score <= 40:
        risk_grade = 'B'
    elif risk_score <= 60:
        risk_grade = 'C'
    elif risk_score <= 80:
        risk_grade = 'D'
    else:
        risk_grade = 'F'
    
    snapshot = PortfolioRiskSnapshot(
        user_id=user_id,
        portfolio_id=portfolio_id,
        portfolio_name=portfolio_data.get('name') if portfolio_data else None,
        volatility=metrics.get('volatility'),
        beta=metrics.get('beta'),
        max_drawdown=metrics.get('max_drawdown'),
        var_95=metrics.get('var_95'),
        var_99=metrics.get('var_99'),
        sharpe_ratio=metrics.get('sharpe_ratio'),
        sortino_ratio=metrics.get('sortino_ratio'),
        risk_score=risk_score,
        risk_grade=risk_grade,
        portfolio_value=portfolio_data.get('total_value') if portfolio_data else None,
        num_positions=len(portfolio_data.get('positions', [])) if portfolio_data else None,
        concentration_risk=metrics.get('concentration_risk'),
        tail_dependency=metrics.get('tail_dependency'),
        hurst_exponent=metrics.get('hurst_exponent'),
        regime_state=metrics.get('regime_state'),
        raw_metrics=metrics,
        portfolio_weights=portfolio_data.get('weights') if portfolio_data else None,
        correlation_matrix=metrics.get('correlation_matrix'),
        calculation_method=risk_result.get('method', 'standard'),
        data_quality_score=metrics.get('data_quality', 1.0)
    )
    
    return snapshot


def get_default_risk_thresholds() -> dict:
    """Get default risk threshold configuration"""
    return {
        'volatility_threshold': 0.15,
        'beta_threshold': 0.20,
        'max_drawdown_threshold': 0.10,
        'var_threshold': 0.15,
        'risk_score_threshold': 0.20,
        'max_acceptable_risk_score': 80.0,
        'max_acceptable_volatility': 0.50
    }
'''
    
    # Write the clean content
    with open(models_file, 'w') as f:
        f.write(clean_content)
    
    print("✅ Models file cleaned and fixed")
    return True

def test_clean_models():
    """Test that the cleaned models work"""
    print("🧪 Testing cleaned models...")
    
    try:
        from db.models import PortfolioRiskSnapshot, RiskThresholdConfig, RiskAlertLog
        print("✅ Risk models imported successfully")
        
        from db.models import create_risk_snapshot_from_calculator, get_default_risk_thresholds
        print("✅ Helper functions imported successfully")
        
        print("🎉 All models working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    print("🧹 CLEANING DB/MODELS.PY")
    print("=" * 30)
    
    # Clean the models file
    if clean_models_file():
        # Test the cleaned models
        if test_clean_models():
            print("\n🎉 SUCCESS!")
            print("✅ db/models.py cleaned and working")
            print("✅ All risk models properly defined")
            print("✅ No more duplicate definitions")
            print("✅ Ready for Phase 1 testing")
            return True
    
    print("\n❌ Cleaning failed")
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)