# api/routes/market_intelligence.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional
import logging
from datetime import datetime

from api.schemas import MarketIntelligenceRequest, MarketIntelligenceResponse
from mcp.schemas import JobRequest
from services.mcp_client import get_mcp_client
from services.proactive_monitor import get_proactive_monitor, AlertPriority, AlertType

# from services.auth import get_current_user  # Uncomment when auth is available

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market-intelligence", tags=["Market Intelligence"])

# Mock auth dependency for testing
async def get_current_user():
    return {"user_id": "test_user_123"}

@router.post("/portfolios/{portfolio_id}/intelligence", response_model=Dict)
async def get_market_intelligence(
    portfolio_id: str,
    request: Dict,  # Simplified for now, will use MarketIntelligenceRequest later
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Request comprehensive market intelligence analysis via MCP"""
    try:
        # Validate portfolio access
        if not await _validate_portfolio_access(portfolio_id, current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        # Create MCP job request
        job_request = JobRequest(
            query=f"Provide comprehensive market intelligence analysis",
            context={
                "portfolio_id": portfolio_id,
                "user_id": current_user["user_id"],
                "time_horizon": request.get("time_horizon", "medium"),
                "include_sentiment": request.get("include_sentiment", True),
                "include_technical": request.get("include_technical", True),
                "focus_sectors": request.get("focus_sectors", [])
            },
            priority=6,
            required_capabilities=["market_intelligence"]
        )
        
        # Submit to MCP
        mcp_client = await get_mcp_client()
        job_response = await mcp_client.submit_job(job_request)
        
        return {
            "job_id": job_response.job_id,
            "status": job_response.status.value,
            "message": "Market intelligence analysis started",
            "estimated_completion": job_response.estimated_completion_time,
            "portfolio_id": portfolio_id
        }
        
    except Exception as e:
        logger.error(f"Error requesting market intelligence: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to request market intelligence")

async def _validate_portfolio_access(portfolio_id: str, user_id: str) -> bool:
    """Validate that user has access to the specified portfolio"""
    # Mock validation - replace with actual database check
    return True

@router.get("/real-time-data")
async def get_real_time_data(
    symbols: Optional[List[str]] = None,
    current_user = Depends(get_current_user)
):
    """Get real-time market data for specified symbols"""
    try:
        # Create MCP job for real-time data
        job_request = JobRequest(
            query=f"Fetch real-time market data for symbols: {symbols or ['major indices']}",
            context={
                "user_id": current_user["user_id"],
                "symbols": symbols,
                "data_type": "real_time"
            },
            priority=8,  # High priority for real-time data
            timeout_seconds=30,
            required_capabilities=["real_time_data"]
        )
        
        mcp_client = await get_mcp_client()
        
        # Use submit_and_wait for real-time data (quick response expected)
        job_response = await mcp_client.submit_and_wait(job_request, max_wait_time=30)
        
        return {
            "data": job_response.result,
            "timestamp": datetime.utcnow().isoformat(),
            "data_freshness": "real_time"
        }
        
    except Exception as e:
        logger.error(f"Error fetching real-time data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch real-time data")

@router.get("/regime-analysis")
async def get_regime_analysis(current_user = Depends(get_current_user)):
    """Get current market regime analysis"""
    try:
        job_request = JobRequest(
            query="Analyze current market regime and forecast",
            context={"user_id": current_user["user_id"]},
            priority=7,
            required_capabilities=["regime_detection"]
        )
        
        mcp_client = await get_mcp_client()
        job_response = await mcp_client.submit_job(job_request)
        
        return {
            "job_id": job_response.job_id,
            "status": job_response.status.value,
            "message": "Regime analysis started"
        }
        
    except Exception as e:
        logger.error(f"Error requesting regime analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to request regime analysis")

@router.post("/portfolios/{portfolio_id}/timing-signals")
async def get_timing_signals(
    portfolio_id: str,
    current_user = Depends(get_current_user)
):
    """Get market timing signals for portfolio"""
    try:
        # Validate portfolio access
        if not await _validate_portfolio_access(portfolio_id, current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        job_request = JobRequest(
            query="Generate market timing signals and recommendations",
            context={
                "portfolio_id": portfolio_id,
                "user_id": current_user["user_id"],
                "analysis_type": "timing_signals"
            },
            priority=7,
            required_capabilities=["market_timing"]
        )
        
        mcp_client = await get_mcp_client()
        job_response = await mcp_client.submit_job(job_request)
        
        return {
            "job_id": job_response.job_id,
            "status": job_response.status.value,
            "message": "Timing signal analysis started",
            "portfolio_id": portfolio_id
        }
        
    except Exception as e:
        logger.error(f"Error requesting timing signals: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to request timing signals")

@router.post("/portfolios/{portfolio_id}/news-impact")
async def get_news_impact(
    portfolio_id: str,
    current_user = Depends(get_current_user)
):
    """Get news impact analysis for portfolio holdings"""
    try:
        if not await _validate_portfolio_access(portfolio_id, current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        job_request = JobRequest(
            query="Analyze news correlation and impact on portfolio holdings",
            context={
                "portfolio_id": portfolio_id,
                "user_id": current_user["user_id"],
                "analysis_type": "news_correlation"
            },
            priority=6,
            required_capabilities=["news_correlation"]
        )
        
        mcp_client = await get_mcp_client()
        job_response = await mcp_client.submit_job(job_request)
        
        return {
            "job_id": job_response.job_id,
            "status": job_response.status.value,
            "message": "News impact analysis started",
            "portfolio_id": portfolio_id
        }
        
    except Exception as e:
        logger.error(f"Error requesting news impact analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to request news impact analysis")
    


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    current_user = Depends(get_current_user)
):
    """Get status and results of a market intelligence job"""
    try:
        mcp_client = await get_mcp_client()
        job_response = await mcp_client.get_job_status(job_id)
        
        if not job_response:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # TODO: Add job ownership validation in production
        # if not await _validate_job_access(job_id, current_user["user_id"]):
        #     raise HTTPException(status_code=403, detail="Access denied to job")
        
        return {
            "job_id": job_id,
            "status": job_response.status.value,
            "progress": job_response.progress or 0.0,
            "message": job_response.message,
            "result": job_response.result,
            "agents_involved": job_response.agents_involved or [],
            "created_at": datetime.utcnow().isoformat(),  # Would use actual timestamp
            "completed": job_response.status.value in ["completed", "failed"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job status")

@router.get("/portfolios/{portfolio_id}/dashboard")
async def get_intelligence_dashboard(
    portfolio_id: str,
    current_user = Depends(get_current_user)
):
    """Get comprehensive market intelligence dashboard for portfolio"""
    try:
        if not await _validate_portfolio_access(portfolio_id, current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        # Submit comprehensive intelligence job
        job_request = JobRequest(
            query="Generate comprehensive market intelligence dashboard",
            context={
                "portfolio_id": portfolio_id,
                "user_id": current_user["user_id"],
                "dashboard_mode": True,
                "include_all_components": True
            },
            priority=8,  # High priority for dashboard
            required_capabilities=["market_intelligence", "regime_detection", "news_correlation", "market_timing"]
        )
        
        mcp_client = await get_mcp_client()
        job_response = await mcp_client.submit_job(job_request)
        
        return {
            "job_id": job_response.job_id,
            "status": job_response.status.value,
            "message": "Intelligence dashboard generation started",
            "portfolio_id": portfolio_id,
            "estimated_completion": job_response.estimated_completion_time,
            "dashboard_components": [
                "real_time_data",
                "market_regime", 
                "timing_signals",
                "news_impact",
                "recommendations"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error requesting intelligence dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to request intelligence dashboard")

@router.get("/health")
async def market_intelligence_health():
    """Health check for market intelligence service"""
    try:
        mcp_client = await get_mcp_client()
        mcp_health = await mcp_client.health_check()
        
        # Check if market intelligence agents are available
        agents = await mcp_client.list_agents()
        market_agents = [
            agent for agent in agents 
            if any(cap in ["market_intelligence", "real_time_data", "regime_detection"] 
                   for cap in agent.get("capabilities", []))
        ]
        
        return {
            "status": "healthy",
            "mcp_status": mcp_health.status,
            "available_market_agents": len(market_agents),
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "real_time_data": "operational",
                "regime_detection": "operational", 
                "news_correlation": "operational",
                "market_timing": "operational"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Additional helper endpoints
@router.get("/available-symbols")
async def get_available_symbols():
    """Get list of symbols available for real-time data"""
    return {
        "major_indices": ["SPY", "QQQ", "IWM", "DIA"],
        "volatility": ["VIX", "VIX9D", "VXST"],
        "sectors": ["XLK", "XLF", "XLE", "XLV", "XLI"],
        "commodities": ["GLD", "SLV", "USO"],
        "bonds": ["TLT", "IEF", "SHY"],
        "currencies": ["UUP", "FXE", "FXY"]
    }

@router.get("/market-status")
async def get_market_status():
    """Get current market trading status"""
    now = datetime.utcnow()
    
    # Simplified market hours check (US Eastern Time)
    hour = now.hour
    weekday = now.weekday()
    
    # Basic market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    # This is simplified - real implementation would handle holidays, etc.
    if weekday < 5 and 13 <= hour < 21:  # UTC hours for 9:30 AM - 4:00 PM ET
        market_status = "open"
    elif weekday < 5 and (9 <= hour < 13 or 21 <= hour < 24):
        market_status = "pre_post_market"
    else:
        market_status = "closed"
    
    return {
        "market_status": market_status,
        "timestamp": now.isoformat(),
        "next_open": "Next weekday 9:30 AM ET" if market_status == "closed" else None,
        "data_availability": "real_time" if market_status == "open" else "delayed"
    }

@router.post("/portfolios/{portfolio_id}/monitoring/start")
async def start_monitoring(
    portfolio_id: str,
    current_user = Depends(get_current_user)
):
    """Start proactive monitoring for a portfolio"""
    try:
        if not await _validate_portfolio_access(portfolio_id, current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        monitor = await get_proactive_monitor()
        result = await monitor.start_portfolio_monitoring(portfolio_id, current_user["user_id"])
        
        return {
            **result,
            "message": f"Proactive monitoring {'already active' if result['status'] == 'already_active' else 'started'} for portfolio {portfolio_id}",
            "monitoring_features": [
                "Risk threshold monitoring",
                "Market regime change detection", 
                "News impact analysis",
                "Market timing signal alerts"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start monitoring")

@router.post("/portfolios/{portfolio_id}/monitoring/stop")
async def stop_monitoring(
    portfolio_id: str,
    current_user = Depends(get_current_user)
):
    """Stop proactive monitoring for a portfolio"""
    try:
        if not await _validate_portfolio_access(portfolio_id, current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        monitor = await get_proactive_monitor()
        result = await monitor.stop_portfolio_monitoring(portfolio_id)
        
        return {
            **result,
            "message": f"Monitoring {'stopped' if result['status'] == 'stopped' else 'was not active'} for portfolio {portfolio_id}"
        }
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to stop monitoring")

@router.get("/monitoring/status")
async def get_monitoring_status(current_user = Depends(get_current_user)):
    """Get status of all active monitoring"""
    try:
        monitor = await get_proactive_monitor()
        active_monitors = monitor.get_active_monitors()
        stats = monitor.get_monitoring_stats()
        
        return {
            "active_portfolios": active_monitors,
            "monitoring_statistics": stats,
            "system_health": "operational",
            "last_check": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring status")

@router.get("/portfolios/{portfolio_id}/monitoring/summary")
async def get_monitoring_summary(
    portfolio_id: str,
    hours: int = 24,
    current_user = Depends(get_current_user)
):
    """Get monitoring summary for a portfolio"""
    try:
        if not await _validate_portfolio_access(portfolio_id, current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        monitor = await get_proactive_monitor()
        summary = await monitor.get_alert_summary(portfolio_id, hours)
        
        # Check if monitoring is active
        active_monitors = monitor.get_active_monitors()
        is_monitoring = portfolio_id in active_monitors
        
        return {
            **summary,
            "is_monitoring_active": is_monitoring,
            "monitoring_recommendation": "Consider starting monitoring" if not is_monitoring else "Monitoring active"
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring summary")

@router.put("/monitoring/thresholds")
async def update_monitoring_thresholds(
    thresholds: Dict[str, float],
    current_user = Depends(get_current_user)
):
    """Update monitoring thresholds (admin function)"""
    try:
        monitor = await get_proactive_monitor()
        result = await monitor.update_thresholds(thresholds)
        
        return {
            **result,
            "message": f"Updated {len(result['updated_thresholds'])} monitoring thresholds",
            "user_id": current_user["user_id"]
        }
        
    except Exception as e:
        logger.error(f"Error updating thresholds: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update thresholds")
    


@router.get("/portfolios/{portfolio_id}/alerts")
async def get_portfolio_alerts(
    portfolio_id: str,
    priority: Optional[str] = None,
    alert_type: Optional[str] = None,
    limit: int = 50,
    current_user = Depends(get_current_user)
):
    """Get alerts for a portfolio"""
    try:
        if not await _validate_portfolio_access(portfolio_id, current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        monitor = await get_proactive_monitor()
        alerts = monitor.get_portfolio_alerts(portfolio_id, limit)
        
        # Filter by priority if specified
        if priority:
            alerts = [a for a in alerts if a["priority"] == priority.lower()]
        
        # Filter by alert type if specified  
        if alert_type:
            alerts = [a for a in alerts if a["alert_type"] == alert_type.lower()]
        
        # Categorize alerts
        unresolved_alerts = [a for a in alerts if not a["resolved"]]
        recent_alerts = [a for a in alerts if 
                        (datetime.utcnow() - datetime.fromisoformat(a["timestamp"])).total_seconds() < 3600]
        
        return {
            "portfolio_id": portfolio_id,
            "alerts": alerts,
            "total_count": len(alerts),
            "unresolved_count": len(unresolved_alerts),
            "recent_count_1h": len(recent_alerts),
            "available_filters": {
                "priorities": [p.value for p in AlertPriority],
                "alert_types": [t.value for t in AlertType]
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    current_user = Depends(get_current_user)
):
    """Mark an alert as resolved"""
    try:
        monitor = await get_proactive_monitor()
        resolved = await monitor.resolve_alert(alert_id)
        
        if resolved:
            return {
                "alert_id": alert_id,
                "status": "resolved",
                "message": "Alert marked as resolved",
                "resolved_by": current_user["user_id"],
                "resolved_at": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")

@router.get("/alerts/system-wide")
async def get_system_wide_alerts(
    priority: Optional[str] = None,
    hours: int = 24,
    limit: int = 100,
    current_user = Depends(get_current_user)
):
    """Get system-wide alerts across all portfolios (admin view)"""
    try:
        monitor = await get_proactive_monitor()
        stats = monitor.get_monitoring_stats()
        
        # Get recent alerts across all portfolios
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        all_alerts = [
            alert for alert in monitor.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        # Filter by priority if specified
        if priority:
            all_alerts = [a for a in all_alerts if a.priority.value == priority.lower()]
        
        # Sort by timestamp (most recent first) and limit
        all_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        all_alerts = all_alerts[:limit]
        
        # Convert to dict format
        alert_dicts = [
            {
                "alert_id": alert.alert_id,
                "portfolio_id": alert.portfolio_id,
                "alert_type": alert.alert_type.value,
                "priority": alert.priority.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            }
            for alert in all_alerts
        ]
        
        return {
            "alerts": alert_dicts,
            "total_count": len(alert_dicts),
            "time_period_hours": hours,
            "system_statistics": stats,
            "alert_distribution": {
                "by_priority": stats["alerts_by_priority"],
                "active_monitors": stats["active_monitors"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving system-wide alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system alerts")

@router.get("/monitoring/demo")
async def trigger_demo_alerts(
    portfolio_id: str = "demo_portfolio",
    current_user = Depends(get_current_user)
):
    """Trigger demo alerts for testing (development only)"""
    try:
        monitor = await get_proactive_monitor()
        
        # Generate some demo alerts
        demo_alerts = await monitor.generate_proactive_alerts(portfolio_id, {
            "user_id": current_user["user_id"],
            "var_breach": True,
            "current_var": 0.035,
            "correlation_spike": True,
            "avg_correlation": 0.85,
            "high_corr_pairs": ["AAPL-MSFT", "GOOGL-META"]
        })
        
        return {
            "message": f"Generated {len(demo_alerts)} demo alerts",
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type.value,
                    "priority": alert.priority.value,
                    "message": alert.message
                }
                for alert in demo_alerts
            ],
            "portfolio_id": portfolio_id,
            "note": "These are demonstration alerts for testing purposes"
        }
        
    except Exception as e:
        logger.error(f"Error generating demo alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate demo alerts")