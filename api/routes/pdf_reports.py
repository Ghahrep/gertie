# api/routes/pdf_reports.py
"""
PDF Reports API Routes
======================
API endpoints for generating professional PDF reports
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime
import io
import asyncio

from db.session import get_db
from db import crud, models
from api.schemas import User
from api.routes.auth import get_current_user

# Import our PDF services
try:
    from services.enhanced_pdf_service import pdf_service
    from services.portfolio_report_templates import (
        portfolio_report_service, PortfolioReportData
    )
    from services.debate_report_templates import (
        debate_report_service, DebateReportData  
    )
    from services.advanced_pdf_features import (
        advanced_pdf_service, BatchReportRequest, ReportTemplate
    )
    PDF_SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"PDF services not available: {e}")
    PDF_SERVICES_AVAILABLE = False

router = APIRouter(prefix="/reports", tags=["PDF Reports"])

# Request/Response Models
class ReportRequest(BaseModel):
    report_type: str  # "holdings", "performance", "risk", "debate"
    portfolio_id: Optional[int] = None
    debate_id: Optional[str] = None
    include_charts: bool = True
    brand_style: str = "corporate_blue"

class BatchReportRequest(BaseModel):
    portfolio_ids: List[int] = []
    debate_ids: List[str] = []
    report_types: List[str] = ["holdings"]
    output_format: str = "pdf"  # "pdf" or "zip"
    template_style: str = "professional"

class ReportResponse(BaseModel):
    success: bool
    report_id: Optional[str] = None
    message: str
    download_url: Optional[str] = None

# Individual Report Generation Endpoints

@router.post("/portfolio/{portfolio_id}/holdings")
async def generate_holdings_report(
    portfolio_id: int,
    include_charts: bool = Query(True),
    brand_style: str = Query("corporate_blue"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate professional holdings summary report"""
    
    if not PDF_SERVICES_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="PDF services not available. Please install reportlab and matplotlib."
        )
    
    # Verify portfolio ownership
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    try:
        # Prepare portfolio data
        holdings = db.query(models.Holding).filter(
            models.Holding.portfolio_id == portfolio_id
        ).all()
        
        # Calculate total value (mock implementation)
        total_value = sum(h.shares * h.current_price for h in holdings if h.current_price)
        
        context = {
            'total_value': total_value,
            'holdings_with_values': holdings,
            'total_day_change': total_value * 0.015  # Mock 1.5% daily change
        }
        
        # Mock analytics data
        analytics = {
            'summary': f'Portfolio {portfolio.name} contains {len(holdings)} holdings with strong performance characteristics.',
            'data': {
                'performance_stats': {
                    'annualized_return_pct': 12.5,
                    'annualized_volatility_pct': 16.2
                },
                'risk_adjusted_ratios': {
                    'sharpe_ratio': 1.35
                },
                'drawdown_stats': {
                    'max_drawdown_pct': -8.5
                },
                'risk_measures': {
                    '95%': {
                        'var': -0.035,
                        'cvar_expected_shortfall': -0.048
                    }
                }
            }
        }
        
        portfolio_data = PortfolioReportData(portfolio, context, analytics)
        
        # Generate branded report
        if brand_style != "professional":
            pdf_buffer = advanced_pdf_service.generate_branded_report(
                portfolio_data, brand_style
            )
        else:
            pdf_buffer = portfolio_report_service.generate_holdings_summary_report(
                portfolio_data
            )
        
        filename = f"holdings_report_{portfolio.name}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate holdings report: {str(e)}"
        )

@router.post("/portfolio/{portfolio_id}/performance")
async def generate_performance_report(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate performance analytics report"""
    
    if not PDF_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="PDF services not available")
    
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    try:
        # Prepare data (similar to holdings report)
        holdings = db.query(models.Holding).filter(
            models.Holding.portfolio_id == portfolio_id
        ).all()
        
        total_value = sum(h.shares * h.current_price for h in holdings if h.current_price)
        
        context = {'total_value': total_value, 'holdings_with_values': holdings}
        analytics = {
            'summary': f'Performance analysis for {portfolio.name} showing strong risk-adjusted returns.',
            'data': {
                'performance_stats': {
                    'annualized_return_pct': 12.5,
                    'annualized_volatility_pct': 16.2,
                    'ytd_return_pct': 8.7,
                    'alpha': 2.3,
                    'beta': 1.05
                },
                'risk_adjusted_ratios': {
                    'sharpe_ratio': 1.35,
                    'sortino_ratio': 1.68,
                    'information_ratio': 0.45
                }
            }
        }
        
        portfolio_data = PortfolioReportData(portfolio, context, analytics)
        pdf_buffer = portfolio_report_service.generate_performance_analytics_report(portfolio_data)
        
        filename = f"performance_report_{portfolio.name}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf", 
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance report failed: {str(e)}")

@router.post("/portfolio/{portfolio_id}/risk")
async def generate_risk_report(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate risk assessment report"""
    
    if not PDF_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="PDF services not available")
    
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    try:
        holdings = db.query(models.Holding).filter(
            models.Holding.portfolio_id == portfolio_id
        ).all()
        
        total_value = sum(h.shares * h.current_price for h in holdings if h.current_price)
        
        context = {'total_value': total_value, 'holdings_with_values': holdings}
        analytics = {
            'summary': f'Risk assessment for {portfolio.name} showing moderate risk profile with effective diversification.',
            'data': {
                'performance_stats': {
                    'annualized_volatility_pct': 16.2,
                    'beta': 1.05
                },
                'risk_measures': {
                    '95%': {
                        'var': -0.035,
                        'cvar_expected_shortfall': -0.048
                    },
                    '99%': {
                        'var': -0.052,
                        'cvar_expected_shortfall': -0.067
                    }
                },
                'drawdown_stats': {
                    'max_drawdown_pct': -8.5,
                    'max_drawdown_duration_days': 42
                }
            }
        }
        
        portfolio_data = PortfolioReportData(portfolio, context, analytics)
        pdf_buffer = portfolio_report_service.generate_risk_assessment_report(portfolio_data)
        
        filename = f"risk_report_{portfolio.name}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk report failed: {str(e)}")

@router.post("/debate/{debate_id}/summary")
async def generate_debate_report(
    debate_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate multi-agent debate summary report"""
    
    if not PDF_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="PDF services not available")
    
    try:
        # Find debate in database
        debate = db.query(models.Debate).filter(
            models.Debate.id == debate_id,
            models.Debate.user_id == current_user.id
        ).first()
        
        if not debate:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        # Get debate participants and messages
        participants = db.query(models.DebateParticipant).filter(
            models.DebateParticipant.debate_id == debate_id
        ).all()
        
        messages = db.query(models.DebateMessage).filter(
            models.DebateMessage.debate_id == debate_id
        ).all()
        
        consensus_items = db.query(models.ConsensusItem).filter(
            models.ConsensusItem.debate_id == debate_id
        ).all()
        
        # Get analytics if available
        analytics = db.query(models.DebateAnalytics).filter(
            models.DebateAnalytics.debate_id == debate_id
        ).first()
        
        debate_data = DebateReportData(debate, participants, messages, consensus_items, analytics)
        pdf_buffer = debate_report_service.generate_debate_summary_report(debate_data)
        
        filename = f"debate_report_{debate_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debate report failed: {str(e)}")

@router.post("/portfolio/{portfolio_id}/dashboard")
async def generate_dashboard_report(
    portfolio_id: int,
    include_ai_insights: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate comprehensive dashboard-style report"""
    
    if not PDF_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="PDF services not available")
    
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.user_id == current_user.id
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    try:
        holdings = db.query(models.Holding).filter(
            models.Holding.portfolio_id == portfolio_id
        ).all()
        
        total_value = sum(h.shares * h.current_price for h in holdings if h.current_price)
        
        context = {'total_value': total_value, 'holdings_with_values': holdings}
        analytics = {
            'summary': f'Dashboard overview for {portfolio.name}',
            'data': {
                'performance_stats': {'annualized_return_pct': 12.5}
            }
        }
        
        portfolio_data = PortfolioReportData(portfolio, context, analytics)
        
        # Optionally include AI debate insights
        debate_data = None
        if include_ai_insights:
            # Find recent debate for this user (mock implementation)
            recent_debate = db.query(models.Debate).filter(
                models.Debate.user_id == current_user.id
            ).order_by(models.Debate.created_at.desc()).first()
            
            if recent_debate:
                participants = db.query(models.DebateParticipant).filter(
                    models.DebateParticipant.debate_id == recent_debate.id
                ).all()
                debate_data = DebateReportData(recent_debate, participants, [], [])
        
        pdf_buffer = advanced_pdf_service.create_comprehensive_dashboard_report(
            portfolio_data, debate_data
        )
        
        filename = f"dashboard_{portfolio.name}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard report failed: {str(e)}")

# Batch Report Generation

@router.post("/batch")
async def generate_batch_reports(
    request: BatchReportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate multiple reports in batch"""
    
    if not PDF_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="PDF services not available")
    
    try:
        # Verify user owns all requested portfolios
        user_portfolios = db.query(models.Portfolio).filter(
            models.Portfolio.id.in_(request.portfolio_ids),
            models.Portfolio.user_id == current_user.id
        ).all()
        
        if len(user_portfolios) != len(request.portfolio_ids):
            raise HTTPException(status_code=404, detail="Some portfolios not found")
        
        # Verify user owns all requested debates
        user_debates = db.query(models.Debate).filter(
            models.Debate.id.in_(request.debate_ids),
            models.Debate.user_id == current_user.id
        ).all()
        
        if len(user_debates) != len(request.debate_ids):
            raise HTTPException(status_code=404, detail="Some debates not found")
        
        # Create batch request
        from services.advanced_pdf_features import BatchReportRequest as ServiceBatchRequest
        
        service_request = ServiceBatchRequest(
            report_type="mixed",
            portfolios=user_portfolios,
            debates=user_debates,
            output_format=request.output_format,
            template_style=request.template_style
        )
        
        # Start batch processing
        batch_result = await advanced_pdf_service.generate_batch_reports(service_request)
        
        return {
            "success": True,
            "batch_id": batch_result["batch_id"],
            "message": batch_result["summary"],
            "status_url": f"/api/v1/reports/batch/{batch_result['batch_id']}/status"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@router.get("/batch/{batch_id}/status")
async def get_batch_status(
    batch_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of batch report generation"""
    
    if not PDF_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="PDF services not available")
    
    status = advanced_pdf_service.get_batch_status(batch_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    return {
        "success": True,
        "batch_status": status
    }

@router.get("/batch/{batch_id}/download") 
async def download_batch_reports(
    batch_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download completed batch reports"""
    
    if not PDF_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="PDF services not available")
    
    status = advanced_pdf_service.get_batch_status(batch_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Batch job not completed yet")
    
    output_buffer = status.get("output")
    if not output_buffer:
        raise HTTPException(status_code=404, detail="Batch output not available")
    
    # Determine media type and filename based on format
    if "zip" in status.get("output_format", "pdf"):
        media_type = "application/zip"
        filename = f"batch_reports_{batch_id}.zip"
    else:
        media_type = "application/pdf"
        filename = f"batch_reports_{batch_id}.pdf"
    
    return StreamingResponse(
        output_buffer,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Health Check

@router.get("/health")
async def reports_health_check():
    """Check PDF services health"""
    
    return {
        "status": "healthy" if PDF_SERVICES_AVAILABLE else "unavailable",
        "pdf_services": PDF_SERVICES_AVAILABLE,
        "services": {
            "enhanced_pdf_service": pdf_service is not None,
            "portfolio_templates": portfolio_report_service is not None,
            "debate_templates": debate_report_service is not None, 
            "advanced_features": advanced_pdf_service is not None
        },
        "timestamp": datetime.now().isoformat()
    }