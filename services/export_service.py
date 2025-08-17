# services/export_service.py

import io
import csv
from datetime import datetime
from typing import Dict, Any, Optional

# The reportlab library is needed for PDF generation
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
except ImportError:
    print("Warning: reportlab not available. PDF export will fail. Install with: pip install reportlab")
    # Define dummy classes if reportlab is not installed to avoid startup errors
    SimpleDocTemplate = None

async def generate_holdings_csv(context: Dict[str, Any]) -> io.StringIO:
    """Generates a CSV of portfolio holdings in memory."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Ticker', 'Shares', 'Current Price', 'Market Value', 'Day Change', 'Day Change %'])
    
    holdings = context.get("holdings_with_values", [])
    if not holdings:
        writer.writerow(['No holdings found in this portfolio.'])
        return output

    for holding in holdings:
        writer.writerow([
            holding.asset.ticker,
            holding.shares,
            f"{holding.current_price:.2f}",
            f"{holding.market_value:.2f}",
            f"{holding.day_change * holding.shares:.2f}",
            f"{holding.day_change_percent:.2f}%"
        ])
    
    output.seek(0)
    return output

async def generate_pdf_report(
    portfolio: Any, 
    context: Dict[str, Any], 
    analytics: Dict[str, Any]
) -> io.BytesIO:
    """Generates a professional PDF report for the portfolio."""
    if SimpleDocTemplate is None:
        raise ImportError("The 'reportlab' library is required for PDF generation.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, rightMargin=inch/2, leftMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"Portfolio Analysis Report: {portfolio.name}", styles['h1']))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.25*inch))

    # AI Summary
    story.append(Paragraph("AI-Generated Summary", styles['h2']))
    summary_text = analytics.get("summary", "No summary available.").replace('\n', '<br/>')
    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 0.25*inch))

    # Key Metrics Table
    story.append(Paragraph("Key Performance & Risk Metrics", styles['h2']))
    
    analytics_data = analytics.get("data", {})
    perf_stats = analytics_data.get("performance_stats", {})
    risk_ratios = analytics_data.get("risk_adjusted_ratios", {})
    drawdown = analytics_data.get("drawdown_stats", {})
    var95 = analytics_data.get("risk_measures", {}).get("95%", {})

    data = [
        ['Metric', 'Value'],
        ['Annualized Return', f"{perf_stats.get('annualized_return_pct', 0):.2f}%"],
        ['Annualized Volatility', f"{perf_stats.get('annualized_volatility_pct', 0):.2f}%"],
        ['Sharpe Ratio', f"{risk_ratios.get('sharpe_ratio', 0):.3f}"],
        ['Max Drawdown', f"{drawdown.get('max_drawdown_pct', 0):.2f}%"],
        ['95% Value-at-Risk (VaR)', f"{var95.get('var', 0) * 100:.2f}%"],
        ['95% Expected Shortfall (CVaR)', f"{var95.get('cvar_expected_shortfall', 0) * 100:.2f}%"],
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.25*inch))

    doc.build(story)
    buffer.seek(0)
    return buffer
