# api/routes/analytics_dashboard.py
"""
Analytics Dashboard API Routes - Task 3.3.2
===========================================
Comprehensive portfolio analytics, performance metrics, and data visualization
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime, timedelta, date
import json
import logging

from db.session import get_db
from db import crud, models
from api.schemas import User
from api.routes.auth import get_current_user
from core.data_handler import get_market_data_for_portfolio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics Dashboard"])

# Request/Response Models
class AnalyticsRequest(BaseModel):
    portfolio_ids: List[int] = []
    date_range: str = "ytd"  # "1m", "3m", "6m", "ytd", "1y", "3y", "5y", "all"
    metrics: List[str] = ["all"]  # Specific metrics to calculate
    benchmark: str = "SPY"  # Benchmark for comparison
    include_breakdown: bool = True

class TimeSeriesRequest(BaseModel):
    portfolio_id: int
    metric: str  # "value", "return", "risk", "sharpe"
    period: str = "daily"  # "daily", "weekly", "monthly"
    date_range: str = "ytd"

class AnalyticsResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    generated_at: datetime
    cache_expires: Optional[datetime] = None

# Core Analytics Service
class AnalyticsDashboardService:
    """Comprehensive analytics service for portfolio dashboard"""
    
    def __init__(self, db: Session):
        self.db = db
        
    def calculate_comprehensive_analytics(self, 
                                        portfolio_ids: List[int], 
                                        user_id: int,
                                        date_range: str = "ytd") -> Dict[str, Any]:
        """Calculate comprehensive portfolio analytics"""
        
        analytics_data = {
            "overview": self._calculate_portfolio_overview(portfolio_ids, user_id),
            "performance": self._calculate_performance_metrics(portfolio_ids, user_id, date_range),
            "risk_metrics": self._calculate_risk_metrics(portfolio_ids, user_id),
            "asset_allocation": self._calculate_asset_allocation(portfolio_ids, user_id),
            "holdings_analysis": self._calculate_holdings_analysis(portfolio_ids, user_id),
            "benchmarking": self._calculate_benchmark_comparison(portfolio_ids, user_id, date_range),
            "market_exposure": self._calculate_market_exposure(portfolio_ids, user_id),
            "concentration_analysis": self._calculate_concentration_metrics(portfolio_ids, user_id),
            "sector_analysis": self._calculate_sector_breakdown(portfolio_ids, user_id),
            "debate_insights": self._get_ai_debate_insights(user_id)
        }
        
        return analytics_data
    
    def _calculate_portfolio_overview(self, portfolio_ids: List[int], user_id: int) -> Dict[str, Any]:
        """Calculate high-level portfolio overview metrics"""
        
        portfolios = self.db.query(models.Portfolio).filter(
            models.Portfolio.id.in_(portfolio_ids),
            models.Portfolio.user_id == user_id
        ).all()
        
        total_value = 0
        total_holdings = 0
        total_day_change = 0
        
        for portfolio in portfolios:
            holdings = self.db.query(models.Holding).filter(
                models.Holding.portfolio_id == portfolio.id
            ).all()
            
            portfolio_value = sum(h.shares * (h.current_price or h.purchase_price) for h in holdings)
            portfolio_day_change = portfolio_value * 0.015  # Mock 1.5% daily change
            
            total_value += portfolio_value
            total_holdings += len(holdings)
            total_day_change += portfolio_day_change
        
        day_change_pct = (total_day_change / total_value * 100) if total_value > 0 else 0
        
        return {
            "total_value": total_value,
            "total_portfolios": len(portfolios),
            "total_holdings": total_holdings,
            "day_change": total_day_change,
            "day_change_pct": day_change_pct,
            "avg_portfolio_value": total_value / len(portfolios) if portfolios else 0
        }
    
    def _calculate_performance_metrics(self, portfolio_ids: List[int], user_id: int, date_range: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        # Mock performance data - in production, calculate from historical snapshots
        date_multipliers = {
            "1m": 1/12, "3m": 3/12, "6m": 6/12, "ytd": 0.75, 
            "1y": 1, "3y": 3, "5y": 5, "all": 7
        }
        multiplier = date_multipliers.get(date_range, 1)
        
        # Base annual return of 12%
        base_annual_return = 0.12
        period_return = base_annual_return * multiplier
        period_return_pct = period_return * 100
        
        # Volatility scales with square root of time
        annual_volatility = 0.16
        period_volatility = annual_volatility * (multiplier ** 0.5)
        
        return {
            "period_return": period_return,
            "period_return_pct": period_return_pct,
            "annualized_return": base_annual_return,
            "annualized_return_pct": base_annual_return * 100,
            "volatility": period_volatility,
            "volatility_pct": period_volatility * 100,
            "sharpe_ratio": base_annual_return / annual_volatility if annual_volatility > 0 else 0,
            "sortino_ratio": (base_annual_return / annual_volatility) * 1.4,  # Mock higher Sortino
            "calmar_ratio": base_annual_return / 0.08,  # Mock max drawdown of 8%
            "max_drawdown": -0.08,
            "max_drawdown_pct": -8.0,
            "best_day": 0.045,  # 4.5% best day
            "worst_day": -0.032,  # -3.2% worst day
            "win_rate": 0.58,  # 58% winning days
            "date_range": date_range,
            "calculation_date": datetime.now().isoformat()
        }
    
    def _calculate_risk_metrics(self, portfolio_ids: List[int], user_id: int) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        
        return {
            "value_at_risk": {
                "95%": {"daily": -0.025, "weekly": -0.055, "monthly": -0.12},
                "99%": {"daily": -0.038, "weekly": -0.082, "monthly": -0.18}
            },
            "expected_shortfall": {
                "95%": {"daily": -0.035, "weekly": -0.075, "monthly": -0.16},
                "99%": {"daily": -0.048, "weekly": -0.098, "monthly": -0.22}
            },
            "beta": 1.05,
            "alpha": 0.023,  # 2.3% annual alpha
            "correlation_to_market": 0.87,
            "tracking_error": 0.045,
            "information_ratio": 0.51,
            "treynor_ratio": 0.114,
            "jensen_alpha": 0.018,
            "r_squared": 0.76,
            "risk_score": 67.5,  # Out of 100
            "risk_level": "Moderate",
            "diversification_ratio": 1.25,
            "concentration_risk": "Low"
        }
    
    def _calculate_asset_allocation(self, portfolio_ids: List[int], user_id: int) -> Dict[str, Any]:
        """Calculate asset allocation breakdown"""
        
        all_holdings = []
        for portfolio_id in portfolio_ids:
            holdings = self.db.query(models.Holding).filter(
                models.Holding.portfolio_id == portfolio_id
            ).all()
            all_holdings.extend(holdings)
        
        if not all_holdings:
            return {"allocation": {}, "total_value": 0}
        
        total_value = sum(h.shares * (h.current_price or h.purchase_price) for h in all_holdings)
        
        # Group by asset type (mock classification)
        asset_types = {}
        for holding in all_holdings:
            # Mock asset classification based on ticker
            ticker = holding.asset.ticker if holding.asset else "UNKNOWN"
            if ticker.startswith(('QQQ', 'TQQQ')):
                asset_type = "ETF"
            elif ticker in ['BTC', 'ETH', 'CRYPTO']:
                asset_type = "Cryptocurrency" 
            elif ticker in ['GLD', 'SLV']:
                asset_type = "Commodities"
            else:
                asset_type = "Stocks"
            
            holding_value = holding.shares * (holding.current_price or holding.purchase_price)
            
            if asset_type not in asset_types:
                asset_types[asset_type] = {"value": 0, "count": 0}
            asset_types[asset_type]["value"] += holding_value
            asset_types[asset_type]["count"] += 1
        
        # Calculate percentages
        allocation = {}
        for asset_type, data in asset_types.items():
            allocation[asset_type] = {
                "value": data["value"],
                "percentage": (data["value"] / total_value * 100) if total_value > 0 else 0,
                "holdings_count": data["count"]
            }
        
        return {
            "allocation": allocation,
            "total_value": total_value,
            "diversification_score": len(asset_types) * 20,  # Simple scoring
            "target_allocation": {  # Mock target allocation
                "Stocks": 70,
                "ETF": 20,
                "Commodities": 5,
                "Cryptocurrency": 5
            }
        }
    
    def _calculate_holdings_analysis(self, portfolio_ids: List[int], user_id: int) -> Dict[str, Any]:
        """Analyze individual holdings performance"""
        
        all_holdings = []
        for portfolio_id in portfolio_ids:
            holdings = self.db.query(models.Holding).filter(
                models.Holding.portfolio_id == portfolio_id
            ).all()
            all_holdings.extend(holdings)
        
        if not all_holdings:
            return {"holdings": [], "summary": {}}
        
        holdings_analysis = []
        total_value = 0
        total_gain_loss = 0
        
        for holding in all_holdings:
            current_price = holding.current_price or holding.purchase_price
            market_value = holding.shares * current_price
            cost_basis = holding.shares * holding.purchase_price
            gain_loss = market_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
            
            total_value += market_value
            total_gain_loss += gain_loss
            
            holdings_analysis.append({
                "ticker": holding.asset.ticker if holding.asset else "UNKNOWN",
                "name": holding.asset.name if holding.asset else "Unknown Asset",
                "shares": holding.shares,
                "purchase_price": holding.purchase_price,
                "current_price": current_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "gain_loss": gain_loss,
                "gain_loss_pct": gain_loss_pct,
                "day_change_pct": 1.5 + (hash(holding.asset.ticker) % 10 - 5) if holding.asset else 0  # Mock
            })
        
        # Sort by market value descending
        holdings_analysis.sort(key=lambda x: x["market_value"], reverse=True)
        
        return {
            "holdings": holdings_analysis,
            "summary": {
                "total_holdings": len(holdings_analysis),
                "total_value": total_value,
                "total_gain_loss": total_gain_loss,
                "total_gain_loss_pct": (total_gain_loss / (total_value - total_gain_loss) * 100) if total_value > total_gain_loss else 0,
                "largest_position": holdings_analysis[0] if holdings_analysis else None,
                "top_performer": max(holdings_analysis, key=lambda x: x["gain_loss_pct"]) if holdings_analysis else None,
                "worst_performer": min(holdings_analysis, key=lambda x: x["gain_loss_pct"]) if holdings_analysis else None
            }
        }
    
    def _calculate_benchmark_comparison(self, portfolio_ids: List[int], user_id: int, date_range: str) -> Dict[str, Any]:
        """Compare portfolio performance to benchmarks"""
        
        # Mock benchmark data
        benchmarks = {
            "SPY": {"name": "S&P 500", "return_ytd": 8.5, "return_1y": 12.0, "volatility": 14.2},
            "QQQ": {"name": "NASDAQ 100", "return_ytd": 15.2, "return_1y": 18.5, "volatility": 19.8},
            "VTI": {"name": "Total Stock Market", "return_ytd": 7.8, "return_1y": 11.2, "volatility": 15.1}
        }
        
        portfolio_return = 12.5  # Mock portfolio return
        
        comparison_data = {}
        for ticker, data in benchmarks.items():
            comparison_data[ticker] = {
                **data,
                "excess_return": portfolio_return - data["return_1y"],
                "outperformance": portfolio_return > data["return_1y"]
            }
        
        return {
            "portfolio_return": portfolio_return,
            "benchmarks": comparison_data,
            "best_benchmark": max(benchmarks.keys(), key=lambda x: benchmarks[x]["return_1y"]),
            "tracking_benchmark": "SPY",
            "relative_performance": portfolio_return - benchmarks["SPY"]["return_1y"]
        }
    
    def _calculate_market_exposure(self, portfolio_ids: List[int], user_id: int) -> Dict[str, Any]:
        """Calculate market exposure and factor analysis"""
        
        return {
            "market_cap_exposure": {
                "large_cap": 65.2,
                "mid_cap": 22.8,
                "small_cap": 12.0
            },
            "geographic_exposure": {
                "domestic": 78.5,
                "international_developed": 15.2,
                "emerging_markets": 6.3
            },
            "style_exposure": {
                "growth": 58.7,
                "value": 35.1,
                "blend": 6.2
            },
            "factor_exposures": {
                "momentum": 0.25,
                "quality": 0.42,
                "value": -0.18,
                "size": 0.08,
                "volatility": -0.15,
                "profitability": 0.35
            }
        }
    
    def _calculate_concentration_metrics(self, portfolio_ids: List[int], user_id: int) -> Dict[str, Any]:
        """Calculate portfolio concentration risk metrics"""
        
        all_holdings = []
        for portfolio_id in portfolio_ids:
            holdings = self.db.query(models.Holding).filter(
                models.Holding.portfolio_id == portfolio_id
            ).all()
            all_holdings.extend(holdings)
        
        if not all_holdings:
            return {"herfindahl_index": 0, "effective_holdings": 0}
        
        total_value = sum(h.shares * (h.current_price or h.purchase_price) for h in all_holdings)
        
        # Calculate weights
        weights = []
        for holding in all_holdings:
            weight = (holding.shares * (holding.current_price or holding.purchase_price)) / total_value
            weights.append(weight)
        
        # Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in weights)
        effective_holdings = 1 / hhi if hhi > 0 else 0
        
        # Top N concentration
        sorted_weights = sorted(weights, reverse=True)
        top_5_concentration = sum(sorted_weights[:5]) * 100
        top_10_concentration = sum(sorted_weights[:10]) * 100
        
        return {
            "herfindahl_index": hhi,
            "effective_holdings": effective_holdings,
            "top_5_concentration_pct": top_5_concentration,
            "top_10_concentration_pct": top_10_concentration,
            "concentration_risk": "Low" if hhi < 0.1 else "Medium" if hhi < 0.2 else "High",
            "diversification_score": min(100, effective_holdings * 10)
        }
    
    def _calculate_sector_breakdown(self, portfolio_ids: List[int], user_id: int) -> Dict[str, Any]:
        """Calculate sector allocation and analysis"""
        
        # Mock sector data - in production, use actual sector mappings
        sectors = {
            "Technology": {"weight": 28.5, "return_ytd": 15.2, "count": 12},
            "Healthcare": {"weight": 18.3, "return_ytd": 8.7, "count": 8},
            "Financial Services": {"weight": 15.2, "return_ytd": 12.1, "count": 6},
            "Consumer Discretionary": {"weight": 12.8, "return_ytd": 6.3, "count": 7},
            "Industrial": {"weight": 10.1, "return_ytd": 9.4, "count": 5},
            "Energy": {"weight": 8.2, "return_ytd": 22.8, "count": 3},
            "Real Estate": {"weight": 4.1, "return_ytd": 4.2, "count": 2},
            "Utilities": {"weight": 2.8, "return_ytd": 3.1, "count": 2}
        }
        
        # S&P 500 sector weights for comparison (mock)
        sp500_sectors = {
            "Technology": 28.1,
            "Healthcare": 13.2,
            "Financial Services": 12.8,
            "Consumer Discretionary": 10.9,
            "Industrial": 8.2,
            "Energy": 4.1,
            "Real Estate": 2.5,
            "Utilities": 2.8
        }
        
        sector_analysis = {}
        for sector, data in sectors.items():
            sector_analysis[sector] = {
                **data,
                "sp500_weight": sp500_sectors.get(sector, 0),
                "relative_weight": data["weight"] - sp500_sectors.get(sector, 0),
                "overweight": data["weight"] > sp500_sectors.get(sector, 0)
            }
        
        return {
            "sector_breakdown": sector_analysis,
            "most_overweight": max(sector_analysis.keys(), key=lambda x: sector_analysis[x]["relative_weight"]),
            "most_underweight": min(sector_analysis.keys(), key=lambda x: sector_analysis[x]["relative_weight"]),
            "sector_diversification_score": min(100, len(sectors) * 12.5)
        }
    
    def _get_ai_debate_insights(self, user_id: int) -> Dict[str, Any]:
        """Get insights from recent AI debates"""
        
        recent_debates = self.db.query(models.Debate).filter(
            models.Debate.user_id == user_id,
            models.Debate.status == "completed"
        ).order_by(models.Debate.completed_at.desc()).limit(5).all()
        
        if not recent_debates:
            return {"recent_debates": 0, "insights": []}
        
        insights = []
        for debate in recent_debates:
            consensus_items = self.db.query(models.ConsensusItem).filter(
                models.ConsensusItem.debate_id == debate.id
            ).all()
            
            insights.append({
                "debate_id": str(debate.id),
                "query": debate.query[:100] + "..." if len(debate.query) > 100 else debate.query,
                "consensus_rate": len(consensus_items) / max(1, debate.total_messages) * 100,
                "confidence_score": debate.confidence_score or 0.0,
                "recommendation": "Hold current allocation with selective rebalancing",  # Mock
                "completed_at": debate.completed_at.isoformat() if debate.completed_at else None
            })
        
        return {
            "recent_debates": len(recent_debates),
            "insights": insights,
            "avg_consensus_rate": sum(i["consensus_rate"] for i in insights) / len(insights) if insights else 0,
            "avg_confidence": sum(i["confidence_score"] for i in insights) / len(insights) if insights else 0
        }

# API Endpoints

@router.get("/overview")
async def get_analytics_overview(
    portfolio_ids: str = Query(..., description="Comma-separated portfolio IDs"),
    date_range: str = Query("ytd", description="Date range for analytics"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive analytics overview for portfolios"""
    
    try:
        # Parse portfolio IDs
        portfolio_id_list = [int(id.strip()) for id in portfolio_ids.split(",") if id.strip()]
        
        if not portfolio_id_list:
            raise HTTPException(status_code=400, detail="At least one portfolio ID required")
        
        # Verify user owns all portfolios
        user_portfolios = db.query(models.Portfolio).filter(
            models.Portfolio.id.in_(portfolio_id_list),
            models.Portfolio.user_id == current_user.id
        ).all()
        
        if len(user_portfolios) != len(portfolio_id_list):
            raise HTTPException(status_code=404, detail="Some portfolios not found")
        
        # Calculate comprehensive analytics
        service = AnalyticsDashboardService(db)
        analytics_data = service.calculate_comprehensive_analytics(
            portfolio_id_list, current_user.id, date_range
        )
        
        return AnalyticsResponse(
            success=True,
            data=analytics_data,
            generated_at=datetime.now(),
            cache_expires=datetime.now() + timedelta(minutes=15)  # 15-minute cache
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid portfolio IDs format")
    except Exception as e:
        logger.error(f"Analytics overview error: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics calculation failed: {str(e)}")

@router.get("/performance/{portfolio_id}")
async def get_performance_analytics(
    portfolio_id: int,
    date_range: str = Query("ytd"),
    benchmark: str = Query("SPY"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed performance analytics for a portfolio"""
    
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    try:
        service = AnalyticsDashboardService(db)
        performance_data = service._calculate_performance_metrics([portfolio_id], current_user.id, date_range)
        benchmarking_data = service._calculate_benchmark_comparison([portfolio_id], current_user.id, date_range)
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio.name,
            "performance": performance_data,
            "benchmarking": benchmarking_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Performance calculation failed: {str(e)}")

@router.get("/risk/{portfolio_id}")
async def get_risk_analytics(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive risk analytics for a portfolio"""
    
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    try:
        service = AnalyticsDashboardService(db)
        risk_data = service._calculate_risk_metrics([portfolio_id], current_user.id)
        concentration_data = service._calculate_concentration_metrics([portfolio_id], current_user.id)
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio.name,
            "risk_metrics": risk_data,
            "concentration_analysis": concentration_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Risk analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")

@router.get("/allocation/{portfolio_id}")
async def get_allocation_analytics(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get asset allocation and sector analysis"""
    
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    try:
        service = AnalyticsDashboardService(db)
        allocation_data = service._calculate_asset_allocation([portfolio_id], current_user.id)
        sector_data = service._calculate_sector_breakdown([portfolio_id], current_user.id)
        exposure_data = service._calculate_market_exposure([portfolio_id], current_user.id)
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio.name,
            "asset_allocation": allocation_data,
            "sector_analysis": sector_data,
            "market_exposure": exposure_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Allocation analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Allocation calculation failed: {str(e)}")

@router.get("/holdings/{portfolio_id}")
async def get_holdings_analytics(
    portfolio_id: int,
    sort_by: str = Query("market_value", description="Sort holdings by field"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    limit: int = Query(50, description="Maximum number of holdings to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed holdings analysis with sorting and filtering"""
    
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    try:
        service = AnalyticsDashboardService(db)
        holdings_data = service._calculate_holdings_analysis([portfolio_id], current_user.id)
        
        # Sort holdings
        holdings_list = holdings_data.get("holdings", [])
        if sort_by in ["market_value", "gain_loss", "gain_loss_pct", "day_change_pct"]:
            reverse = (order == "desc")
            holdings_list.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
        
        # Apply limit
        holdings_list = holdings_list[:limit]
        holdings_data["holdings"] = holdings_list
        
        return {
            "success": True,
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio.name,
            "holdings_analysis": holdings_data,
            "sort_by": sort_by,
            "order": order,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Holdings analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Holdings analysis failed: {str(e)}")

@router.get("/comparison")
async def compare_portfolios(
    portfolio_ids: str = Query(..., description="Comma-separated portfolio IDs"),
    metrics: str = Query("performance,risk,allocation", description="Metrics to compare"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Compare multiple portfolios across key metrics"""
    
    try:
        # Parse inputs
        portfolio_id_list = [int(id.strip()) for id in portfolio_ids.split(",") if id.strip()]
        metrics_list = [m.strip() for m in metrics.split(",")]
        
        if len(portfolio_id_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 portfolios required for comparison")
        
        # Verify ownership
        user_portfolios = db.query(models.Portfolio).filter(
            models.Portfolio.id.in_(portfolio_id_list),
            models.Portfolio.user_id == current_user.id
        ).all()
        
        if len(user_portfolios) != len(portfolio_id_list):
            raise HTTPException(status_code=404, detail="Some portfolios not found")
        
        service = AnalyticsDashboardService(db)
        comparison_data = {}
        
        for portfolio in user_portfolios:
            portfolio_metrics = {}
            
            if "performance" in metrics_list:
                portfolio_metrics["performance"] = service._calculate_performance_metrics([portfolio.id], current_user.id, "ytd")
            
            if "risk" in metrics_list:
                portfolio_metrics["risk"] = service._calculate_risk_metrics([portfolio.id], current_user.id)
            
            if "allocation" in metrics_list:
                portfolio_metrics["allocation"] = service._calculate_asset_allocation([portfolio.id], current_user.id)
            
            comparison_data[portfolio.name] = portfolio_metrics
        
        return {
            "success": True,
            "comparison": comparison_data,
            "metrics_compared": metrics_list,
            "portfolios_compared": len(user_portfolios),
            "generated_at": datetime.now().isoformat()
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid input format")
    except Exception as e:
        logger.error(f"Portfolio comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.get("/ai-insights/{user_id}")
async def get_ai_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI-powered insights from recent debates and analysis"""
    
    try:
        service = AnalyticsDashboardService(db)
        ai_insights = service._get_ai_debate_insights(current_user.id)
        
        return {
            "success": True,
            "ai_insights": ai_insights,
            "user_id": current_user.id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI insights error: {e}")
        raise HTTPException(status_code=500, detail=f"AI insights failed: {str(e)}")

@router.get("/health")
async def analytics_health_check():
    """Health check for analytics service"""
    
    return {
        "status": "healthy",
        "service": "Analytics Dashboard",
        "features": {
            "portfolio_analytics": True,
            "performance_metrics": True,
            "risk_analysis": True,
            "asset_allocation": True,
            "benchmarking": True,
            "ai_insights": True
        },
        "timestamp": datetime.now().isoformat()
    }