# in main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime

# Import your new orchestrator
from agents.orchestrator import FinancialOrchestrator
from api.routes import users, auth, portfolios
from db.session import get_db
from db import crud  # Add this import for crud operations
from api.schemas import User
from core.data_handler import get_market_data_for_portfolio  # Add this import
from api.routes import alerts

# --- API Application Setup ---
app = FastAPI(
    title="Gertie.ai - Multi-Agent Financial Platform",
    description="An API for interacting with a team of specialized financial AI agents.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "null"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include the routers in your main app
app.include_router(users.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(portfolios.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])

# Create a single, long-lived instance of the orchestrator
orchestrator = FinancialOrchestrator()

# --- API Request & Response Models ---
class ChatRequest(BaseModel):
    query: str

# --- MOVED OUTSIDE AND FIXED INDENTATION ---
def format_agent_response_for_frontend(result, agent_type):
    """Format your sophisticated agent responses for simple frontend consumption"""
    
    # Agent display names and specializations
    agent_info = {
        "quantitative": {"name": "Dr. Sarah Chen", "specialization": "Quantitative Analysis & Risk Metrics"},
        "strategy": {"name": "Marcus Rodriguez", "specialization": "Investment Strategy & Portfolio Optimization"}, 
        "risk": {"name": "Elena Volkov", "specialization": "Risk Management & Hedging"},
        "rebalancing": {"name": "Alex Thompson", "specialization": "Portfolio Rebalancing"},
        "regime": {"name": "Dr. James Liu", "specialization": "Market Regime Analysis"},
        "behavioral": {"name": "Dr. Maya Patel", "specialization": "Behavioral Finance"},
        "scenario": {"name": "Robert Kim", "specialization": "Scenario Analysis & Stress Testing"},
        "tutor": {"name": "Professor Williams", "specialization": "Financial Education"}
    }
    
    agent = agent_info.get(agent_type, {"name": "Financial Advisor", "specialization": "General Analysis"})
    
    # Handle both success and error cases
    if not result or not result.get("success", True):
        error_msg = result.get("error", "Analysis could not be completed") if result else "No response from agent"
        
        # Provide helpful suggestions based on agent type
        suggestions = {
            "strategy": "Try asking: 'recommend portfolio rebalancing' or 'design momentum strategy for AAPL MSFT GOOG'",
            "risk": "Try asking: 'analyze portfolio risk' or 'calculate hedge ratios'",
            "quantitative": "Try asking: 'calculate risk metrics' or 'perform stress test'"
        }
        
        return {
            "agent": agent["name"],
            "agent_type": agent_type,
            "specialization": agent["specialization"],
            "message": f"I encountered an issue: {error_msg}. {suggestions.get(agent_type, 'Please rephrase your question.')}",
            "metrics": {},
            "recommendations": ["Try rephrasing your question", "Use specific financial terms", "Ask about portfolio analysis"],
            "confidence": 0.1,
            "timestamp": datetime.now().isoformat(),
            "error": True
        }
    
    # Extract metrics from your sophisticated agent response
    metrics = {}
    data = result.get("data", {}) if result else {}
    
    # Extract performance and risk metrics
    if "risk_adjusted_ratios" in data:
        ratios = data["risk_adjusted_ratios"]
        if "sharpe_ratio" in ratios:
            metrics["Sharpe Ratio"] = f"{ratios['sharpe_ratio']:.3f}"
        if "sortino_ratio" in ratios:
            metrics["Sortino Ratio"] = f"{ratios['sortino_ratio']:.3f}"
        if "calmar_ratio" in ratios:
            metrics["Calmar Ratio"] = f"{ratios['calmar_ratio']:.3f}"
    
    if "performance_stats" in data:
        perf = data["performance_stats"]
        if "annualized_return_pct" in perf:
            metrics["Annual Return"] = f"{perf['annualized_return_pct']:.1f}%"
        if "annualized_volatility_pct" in perf:
            metrics["Volatility"] = f"{perf['annualized_volatility_pct']:.1f}%"
    
    if "risk_measures" in data:
        risk = data["risk_measures"]
        if "95%" in risk:
            metrics["VaR 95%"] = f"{abs(risk['95%']['var']) * 100:.2f}%"
        if "99%" in risk:
            metrics["CVaR 99%"] = f"{risk['99%']['cvar_expected_shortfall'] * 100:.2f}%"
    
    if "drawdown_stats" in data:
        dd = data["drawdown_stats"]
        if "max_drawdown_pct" in dd:
            metrics["Max Drawdown"] = f"{dd['max_drawdown_pct']:.1f}%"
    
    # Extract recommendations based on agent type and response
    recommendations = []
    summary = result.get("summary", "") if result else ""
    agent_used = result.get("agent_used", "")
    
    if agent_type == "quantitative" or "QuantitativeAnalyst" in agent_used:
        recommendations = [
            "Monitor your Sharpe ratio - current level shows good risk-adjusted returns",
            "Consider stress testing with tail risk analysis",
            "Review portfolio correlation and diversification"
        ]
    elif agent_type == "strategy" or "Strategy" in agent_used:
        if "rebalance" in summary.lower():
            recommendations = [
                "Implement suggested portfolio rebalancing",
                "Consider HERC optimization for better diversification",
                "Monitor regime changes for strategy adjustment"
            ]
        else:
            recommendations = [
                "Consider portfolio optimization using modern techniques",
                "Evaluate momentum vs mean-reversion strategies",
                "Review asset allocation targets"
            ]
    elif agent_type == "risk" or "risk" in summary.lower():
        recommendations = [
            "Implement protective hedging strategies",
            "Monitor Value at Risk thresholds", 
            "Consider volatility targeting approach"
        ]
    else:
        recommendations = [
            "Review analysis recommendations carefully",
            "Consider implementing suggested changes gradually",
            "Monitor portfolio metrics regularly"
        ]
    
    # Extract optimization results if available
    if "optimization_results" in data:
        opt = data["optimization_results"]
        if "optimal_weights" in opt:
            metrics["Optimization"] = "Portfolio weights calculated"
    
    if "trade_plan" in data and data["trade_plan"]:
        trades = data["trade_plan"]
        recommendations.append(f"Execute {len(trades)} recommended trades")
    
    return {
        "agent": agent["name"],
        "agent_type": agent_type,
        "specialization": agent["specialization"],
        "message": result.get("summary", "Analysis completed successfully.") if result else "Analysis in progress...",
        "metrics": metrics,
        "recommendations": recommendations,
        "confidence": 0.92 if result.get("success", True) else 0.3,
        "timestamp": datetime.now().isoformat(),
        "full_data": data  # Include full data for advanced frontend features
    }

# --- API Endpoint ---
@app.post("/api/v1/chat", tags=["Agent Interaction"])
def chat_with_agent_team(
    request: ChatRequest,
    agent_type: Optional[str] = None,  # Add this parameter
    db: Session = Depends(get_db), 
    current_user: User = Depends(auth.get_current_user)
):
    """Enhanced chat with agent-specific routing for frontend"""
    
    # Map frontend agent names to your sophisticated backend agents
    agent_mapping = {
        "quantitative": "QuantitativeAnalystAgent",
        "dr_sarah": "QuantitativeAnalystAgent", 
        "strategy": "StrategyArchitectAgent",
        "marcus": "StrategyArchitectAgent",
        "risk": "HedgingStrategistAgent", 
        "elena": "HedgingStrategistAgent",
        "rebalancing": "StrategyRebalancingAgent",
        "backtesting": "StrategyBacktesterAgent",
        "regime": "RegimeForecastingAgent",
        "behavioral": "BehavioralFinanceAgent",
        "scenario": "ScenarioSimulationAgent",
        "tutor": "FinancialTutorAgent"
    }
    
    # If specific agent requested, route directly to that specialized agent
    if agent_type and agent_type in agent_mapping:
        target_agent_name = agent_mapping[agent_type]
        
        # Get portfolio context using your existing sophisticated data handler
        user_portfolios = crud.get_user_portfolios(db=db, user_id=current_user.id)
        portfolio_context = {}
        if user_portfolios:
            portfolio_context = get_market_data_for_portfolio(user_portfolios[0].holdings)
        
        # Enhance query for strategy agent
        enhanced_query = request.query
        if agent_type == "strategy" and not any(word in request.query.lower() for word in ["rebalance", "momentum", "mean-reversion"]):
            # Add strategy-specific context
            tickers = [h.asset.ticker for h in user_portfolios[0].holdings if h.asset] if user_portfolios else []
            if tickers:
                enhanced_query = f"recommend portfolio rebalancing strategy for my holdings: {', '.join(tickers[:5])}"
            else:
                enhanced_query = "recommend portfolio optimization and rebalancing strategy"
        
        # Run your sophisticated agent
        result = orchestrator.roster[target_agent_name].run(
            enhanced_query, 
            context=portfolio_context
        )
        
        # Format your rich agent response for frontend consumption
        return format_agent_response_for_frontend(result, agent_type)
    
    # Otherwise use your existing sophisticated orchestrator
    result_data = orchestrator.route_query(
        user_query=request.query, 
        db_session=db, 
        current_user=current_user
    )
    return {"data": result_data}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Gertie.ai API. Please use the /docs endpoint to interact."}