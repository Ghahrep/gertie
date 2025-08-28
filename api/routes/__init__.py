# api/routes/__init__.py

from . import (
    agent_debates,
    ai_analysis,
    alerts,
    analytics_dashboard,  # Note: your file is named analytic_dashboard.py (singular)
    auth,
    contextual_chat,
    csv_import,
    csv_service,
    debate_stream,
    debates,
    export_routes,
    market_intelligence,
    pdf_reports,
    portfolios,
    smart_suggestions,
    users,
    websocket_endpoints
)

__all__ = [
    "agent_debates",
    "ai_analysis", 
    "alerts",
    "analytics_dashboard",
    "auth",
    "contextual_chat",
    "csv_import",
    "csv_service",
    "debate_stream",
    "debates",
    "export_routes",
    "market_intelligence",
    "pdf_reports", 
    "portfolios",
    "smart_suggestions",
    "users",
    "websocket_endpoints"
]