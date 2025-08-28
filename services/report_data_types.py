# services/report_data_types.py
"""
Shared data types for report generation
=======================================
Common data structures used across all report templates to avoid circular imports
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class PortfolioReportData:
    """Structured data container for portfolio reports"""
    portfolio: Any
    context: Dict[str, Any]
    analytics: Dict[str, Any]
    holdings: List[Any] = None
    risk_metrics: Dict[str, Any] = None
    performance_history: List[Dict] = None
    market_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.holdings is None:
            self.holdings = self.context.get('holdings_with_values', [])
        if self.risk_metrics is None:
            self.risk_metrics = self.analytics.get('data', {}).get('risk_measures', {})
        if self.market_data is None:
            self.market_data = self.context.get('market_data', {})

@dataclass
class DebateReportData:
    """Structured data container for debate reports"""
    debate_topic: str
    participants: List[Dict[str, Any]]
    arguments: Dict[str, List[Dict[str, Any]]]
    context: Dict[str, Any]
    analytics: Dict[str, Any] = None
    summary: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.analytics is None:
            self.analytics = {}
        if self.summary is None:
            self.summary = {}

@dataclass
class BaseReportData:
    """Base class for all report data types"""
    title: str
    created_at: str
    report_type: str
    context: Dict[str, Any]