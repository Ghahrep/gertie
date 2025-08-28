# api/routes/export_routes.py
"""
Export API Routes - Excel, PDF, CSV, and JSON exports
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from typing import Dict, Any
import io

from db.session import get_db
from db.crud import get_portfolio_by_id, get_user_by_id
from services.export_service import (
    generate_excel_report,
    generate_pdf_report,
    generate_holdings_csv,
    generate_json_export,
    EXCEL_AVAILABLE,
    PDF_AVAILABLE
)
#from services.portfolio_analytics_service import PortfolioAnalyticsService
from api.routes.auth import get_current_user

router = APIRouter(prefix="/export", tags=["export"])

@router.get("/portfolios/{portfolio_id}/excel")
async def export_portfolio_excel(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Export portfolio as Excel workbook with multiple sheets."""
    
    if not EXCEL_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Excel export not available. Install openpyxl package."
        )
    
    # Get portfolio
    portfolio = get_portfolio_by_id(db, portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Check ownership
    if portfolio.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # Get analytics data
        analytics_service = PortfolioAnalyticsService()
        context = await analytics_service.get_portfolio_context(db, portfolio_id)
        analytics = await analytics_service.get_comprehensive_analytics(portfolio, context)
        
        # Generate Excel report
        excel_buffer = await generate_excel_report(portfolio, context, analytics)
        
        return Response(
            content=excel_buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=portfolio_{portfolio_id}_report.xlsx"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/portfolios/{portfolio_id}/pdf")
async def export_portfolio_pdf(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Export portfolio as PDF report."""
    
    if not PDF_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="PDF export not available. Install reportlab package."
        )
    
    # Get portfolio
    portfolio = get_portfolio_by_id(db, portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Check ownership
    if portfolio.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # Get analytics data
        analytics_service = PortfolioAnalyticsService()
        context = await analytics_service.get_portfolio_context(db, portfolio_id)
        analytics = await analytics_service.get_comprehensive_analytics(portfolio, context)
        
        # Generate PDF report
        pdf_buffer = await generate_pdf_report(portfolio, context, analytics)
        
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=portfolio_{portfolio_id}_report.pdf"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/portfolios/{portfolio_id}/csv")
async def export_portfolio_csv(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Export portfolio holdings as CSV."""
    
    # Get portfolio
    portfolio = get_portfolio_by_id(db, portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Check ownership
    if portfolio.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # Get portfolio context
        analytics_service = PortfolioAnalyticsService()
        context = await analytics_service.get_portfolio_context(db, portfolio_id)
        
        # Generate CSV
        csv_buffer = await generate_holdings_csv(context)
        
        return Response(
            content=csv_buffer.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=portfolio_{portfolio_id}_holdings.csv"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/portfolios/{portfolio_id}/json")
async def export_portfolio_json(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Export portfolio data as JSON for API integrations."""
    
    # Get portfolio
    portfolio = get_portfolio_by_id(db, portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Check ownership
    if portfolio.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # Get analytics data
        analytics_service = PortfolioAnalyticsService()
        context = await analytics_service.get_portfolio_context(db, portfolio_id)
        analytics = await analytics_service.get_comprehensive_analytics(portfolio, context)
        
        # Generate JSON export
        json_data = await generate_json_export(portfolio, context, analytics)
        
        return Response(
            content=json_data,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=portfolio_{portfolio_id}_data.json"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/portfolios/{portfolio_id}/formats")
async def get_available_export_formats(portfolio_id: int):
    """Get available export formats for this portfolio."""
    
    formats = {
        "csv": {"available": True, "description": "Holdings data as CSV"},
        "json": {"available": True, "description": "Complete portfolio data as JSON"},
        "pdf": {"available": PDF_AVAILABLE, "description": "Professional PDF report"},
        "excel": {"available": EXCEL_AVAILABLE, "description": "Multi-sheet Excel workbook"}
    }
    
    return {
        "portfolio_id": portfolio_id,
        "available_formats": formats,
        "endpoints": {
            "csv": f"/export/portfolios/{portfolio_id}/csv",
            "json": f"/export/portfolios/{portfolio_id}/json", 
            "pdf": f"/export/portfolios/{portfolio_id}/pdf",
            "excel": f"/export/portfolios/{portfolio_id}/excel"
        }
    }

class PortfolioAnalyticsService:
    async def get_portfolio_context(self, db, portfolio_id):
        return {
            "total_value": 100000,
            "holdings_with_values": [],
            "total_day_change": 1500
        }
    
    async def get_comprehensive_analytics(self, portfolio, context):
        return {
            "summary": "Mock analytics data for testing",
            "data": {
                "performance_stats": {
                    "annualized_return_pct": 8.5,
                    "annualized_volatility_pct": 15.2
                },
                "risk_adjusted_ratios": {
                    "sharpe_ratio": 0.56
                },
                "drawdown_stats": {
                    "max_drawdown_pct": -12.3
                },
                "risk_measures": {
                    "95%": {
                        "var": -0.025,
                        "cvar_expected_shortfall": -0.035
                    }
                }
            }
        }