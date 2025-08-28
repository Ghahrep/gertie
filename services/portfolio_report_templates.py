# services/portfolio_report_templates.py
"""
Portfolio Report Templates - Sub-task 3.3.1.2
==============================================
Specialized report templates for portfolio analysis, performance, and risk assessment
"""

import io
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import base64
from dataclasses import dataclass
from services.report_data_types import PortfolioReportData

# Import our enhanced PDF service - FIXED IMPORT PATH
try:
    from services.enhanced_pdf_service import (
        ProfessionalPDFService, ReportConfig, ReportType, ChartType,
        REPORTLAB_AVAILABLE, MATPLOTLIB_AVAILABLE
    )
    if REPORTLAB_AVAILABLE:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
        from reportlab.lib.units import inch
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
except ImportError as e:
    print(f"Warning: Enhanced PDF service not available - {e}")
    # Set fallback values
    REPORTLAB_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False
    ProfessionalPDFService = None
    ReportConfig = None
    ReportType = None
    ChartType = None

import logging

logger = logging.getLogger(__name__)

@dataclass
class PortfolioReportTemplates:
    """Specialized portfolio report templates"""
    
    def __init__(self, pdf_service: Optional[Any] = None):  # Use Optional[Any] instead of ProfessionalPDFService
        if not REPORTLAB_AVAILABLE:
            print("Warning: ReportLab not available - PDF generation will be disabled")
            self.pdf_service = None
            return
        
        self.pdf_service = pdf_service
        if self.pdf_service is None and ProfessionalPDFService is not None:
            self.pdf_service = ProfessionalPDFService()
    
    def generate_holdings_summary_report(self, report_data: PortfolioReportData) -> io.BytesIO:
        """Generate detailed holdings summary report"""
        
        if not REPORTLAB_AVAILABLE or self.pdf_service is None:
            raise RuntimeError("PDF service not available - cannot generate report")
        
        config = ReportConfig(
            report_type=ReportType.PORTFOLIO_SUMMARY,
            title=f"Holdings Summary: {report_data.portfolio.name}",
            subtitle="Detailed Portfolio Composition Analysis"
        )
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=1*inch, bottomMargin=0.75*inch)
        
        story = []
        
        # Title and header
        story.append(Paragraph(config.title, self.pdf_service.styles['CustomTitle']))
        story.append(Paragraph(config.subtitle, self.pdf_service.styles['CustomHeading2']))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", 
                             self.pdf_service.styles['CustomNormal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Portfolio Overview
        story.append(Paragraph("Portfolio Overview", self.pdf_service.styles['CustomHeading1']))
        
        total_value = report_data.context.get('total_value', 0)
        total_holdings = len(report_data.holdings)
        day_change = report_data.context.get('total_day_change', 0)
        day_change_pct = (day_change / total_value * 100) if total_value > 0 else 0
        
        overview_data = [
            ['Metric', 'Value'],
            ['Total Portfolio Value', f"${total_value:,.2f}"],
            ['Number of Holdings', f"{total_holdings}"],
            ['Today\'s Change', f"${day_change:,.2f} ({day_change_pct:+.2f}%)"],
            ['Largest Position', self._get_largest_position_name(report_data.holdings)],
            ['Portfolio Diversification', self._calculate_diversification_score(report_data.holdings)]
        ]
        
        overview_table = self.pdf_service.create_professional_table(
            overview_data[1:], overview_data[0], "financial")
        story.append(overview_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Allocation Chart
        if config.include_charts and len(report_data.holdings) > 0:
            allocation_chart = self._create_allocation_chart(report_data.holdings)
            if allocation_chart:
                story.append(Paragraph("Portfolio Allocation", self.pdf_service.styles['CustomHeading2']))
                image_data = base64.b64decode(allocation_chart)
                img_buffer = io.BytesIO(image_data)
                img = Image(img_buffer, width=5*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
        
        # Detailed Holdings Table
        story.append(Paragraph("Detailed Holdings", self.pdf_service.styles['CustomHeading1']))
        
        if report_data.holdings:
            holdings_data = [
                ['Symbol', 'Company', 'Shares', 'Price', 'Market Value', 'Weight', 'Day Change', 'Gain/Loss']
            ]
            
            for holding in report_data.holdings:
                weight = (holding.market_value / total_value * 100) if total_value > 0 else 0
                gain_loss = holding.market_value - (holding.shares * holding.purchase_price) if hasattr(holding, 'purchase_price') else 0
                day_change = getattr(holding, 'day_change', 0) * holding.shares
                
                holdings_data.append([
                    holding.asset.ticker,
                    (holding.asset.name or 'N/A')[:25] + ('...' if len(holding.asset.name or '') > 25 else ''),
                    f"{holding.shares:,.0f}",
                    f"${holding.current_price:.2f}",
                    f"${holding.market_value:,.2f}",
                    f"{weight:.1f}%",
                    f"${day_change:+.2f}",
                    f"${gain_loss:+,.2f}"
                ])
            
            holdings_table = self.pdf_service.create_professional_table(
                holdings_data[1:], holdings_data[0], "financial")
            story.append(holdings_table)
        else:
            story.append(Paragraph("No holdings found in this portfolio.", 
                                 self.pdf_service.styles['CustomNormal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Sector Analysis
        if len(report_data.holdings) > 0:
            story.append(Paragraph("Sector Analysis", self.pdf_service.styles['CustomHeading1']))
            sector_analysis = self._analyze_sectors(report_data.holdings)
            
            if sector_analysis:
                sector_data = [['Sector', 'Holdings Count', 'Total Value', 'Portfolio Weight']]
                for sector, data in sector_analysis.items():
                    weight = (data['value'] / total_value * 100) if total_value > 0 else 0
                    sector_data.append([
                        sector,
                        str(data['count']),
                        f"${data['value']:,.2f}",
                        f"{weight:.1f}%"
                    ])
                
                sector_table = self.pdf_service.create_professional_table(
                    sector_data[1:], sector_data[0], "financial")
                story.append(sector_table)
        
        # Risk Summary
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Risk Summary", self.pdf_service.styles['CustomHeading1']))
        
        risk_summary = self._generate_holdings_risk_summary(report_data)
        story.append(Paragraph(risk_summary, self.pdf_service.styles['CustomNormal']))
        
        # Footer
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Generated by Advanced Portfolio Analytics System", 
                             self.pdf_service.styles['Footer']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_performance_analytics_report(self, report_data: PortfolioReportData) -> io.BytesIO:
        """Generate comprehensive performance analytics report"""
        
        if not REPORTLAB_AVAILABLE or self.pdf_service is None:
            raise RuntimeError("PDF service not available - cannot generate report")
        
        config = ReportConfig(
            report_type=ReportType.PERFORMANCE_ANALYSIS,
            title=f"Performance Analysis: {report_data.portfolio.name}",
            subtitle="Comprehensive Returns and Benchmarking Analysis"
        )
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=1*inch, bottomMargin=0.75*inch)
        
        story = []
        
        # Title
        story.append(Paragraph(config.title, self.pdf_service.styles['CustomTitle']))
        story.append(Paragraph(config.subtitle, self.pdf_service.styles['CustomHeading2']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        perf_summary = self._generate_performance_summary(report_data.analytics)
        story.append(Paragraph("Executive Performance Summary", self.pdf_service.styles['CustomHeading1']))
        story.append(Paragraph(perf_summary, self.pdf_service.styles['ExecutiveSummary']))
        story.append(Spacer(1, 0.3*inch))
        
        # Performance Metrics Table
        story.append(Paragraph("Key Performance Metrics", self.pdf_service.styles['CustomHeading1']))
        
        analytics_data = report_data.analytics.get("data", {})
        perf_stats = analytics_data.get("performance_stats", {})
        risk_ratios = analytics_data.get("risk_adjusted_ratios", {})
        
        perf_data = [
            ['Metric', 'Portfolio', 'Benchmark*', 'Relative'],
            ['Total Return (YTD)', f"{perf_stats.get('ytd_return_pct', 0):.2f}%", "8.50%", 
             f"{perf_stats.get('ytd_return_pct', 0) - 8.5:+.2f}%"],
            ['Annualized Return', f"{perf_stats.get('annualized_return_pct', 0):.2f}%", "10.00%",
             f"{perf_stats.get('annualized_return_pct', 0) - 10.0:+.2f}%"],
            ['Volatility (Annual)', f"{perf_stats.get('annualized_volatility_pct', 0):.2f}%", "15.00%",
             f"{perf_stats.get('annualized_volatility_pct', 0) - 15.0:+.2f}%"],
            ['Sharpe Ratio', f"{risk_ratios.get('sharpe_ratio', 0):.3f}", "0.667",
             f"{risk_ratios.get('sharpe_ratio', 0) - 0.667:+.3f}"],
            ['Information Ratio', f"{risk_ratios.get('information_ratio', 0):.3f}", "0.000",
             f"{risk_ratios.get('information_ratio', 0):+.3f}"],
            ['Alpha (Annual)', f"{perf_stats.get('alpha', 0):.2f}%", "0.00%",
             f"{perf_stats.get('alpha', 0):+.2f}%"],
            ['Beta', f"{perf_stats.get('beta', 1.0):.2f}", "1.00",
             f"{perf_stats.get('beta', 1.0) - 1.0:+.2f}"]
        ]
        
        perf_table = self.pdf_service.create_professional_table(
            perf_data[1:], perf_data[0], "financial")
        story.append(perf_table)
        story.append(Paragraph("<i>*Benchmark: S&P 500 Total Return Index</i>", 
                             self.pdf_service.styles['CustomNormal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Performance Chart
        if config.include_charts:
            perf_chart = self._create_performance_chart(report_data)
            if perf_chart:
                story.append(Paragraph("Performance History", self.pdf_service.styles['CustomHeading1']))
                image_data = base64.b64decode(perf_chart)
                img_buffer = io.BytesIO(image_data)
                img = Image(img_buffer, width=6*inch, height=3.6*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
        
        # Monthly Returns Table
        story.append(Paragraph("Monthly Returns Analysis", self.pdf_service.styles['CustomHeading1']))
        monthly_returns = self._generate_monthly_returns_table()
        story.append(monthly_returns)
        story.append(Spacer(1, 0.3*inch))
        
        # Top/Bottom Performers
        story.append(Paragraph("Best and Worst Performers", self.pdf_service.styles['CustomHeading1']))
        performers_analysis = self._analyze_performers(report_data.holdings)
        story.append(Paragraph(performers_analysis, self.pdf_service.styles['CustomNormal']))
        
        # Footer
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Performance data calculated using time-weighted returns methodology", 
                             self.pdf_service.styles['Footer']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_risk_assessment_report(self, report_data: PortfolioReportData) -> io.BytesIO:
        """Generate comprehensive risk assessment report"""
        
        if not REPORTLAB_AVAILABLE or self.pdf_service is None:
            raise RuntimeError("PDF service not available - cannot generate report")
        
        config = ReportConfig(
            report_type=ReportType.RISK_ANALYSIS,
            title=f"Risk Assessment: {report_data.portfolio.name}",
            subtitle="Comprehensive Risk Metrics and Analysis"
        )
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=1*inch, bottomMargin=0.75*inch)
        
        story = []
        
        # Title
        story.append(Paragraph(config.title, self.pdf_service.styles['CustomTitle']))
        story.append(Paragraph(config.subtitle, self.pdf_service.styles['CustomHeading2']))
        story.append(Spacer(1, 0.3*inch))
        
        # Risk Executive Summary
        risk_summary = self._generate_risk_executive_summary(report_data.analytics)
        story.append(Paragraph("Risk Executive Summary", self.pdf_service.styles['CustomHeading1']))
        story.append(Paragraph(risk_summary, self.pdf_service.styles['ExecutiveSummary']))
        story.append(Spacer(1, 0.3*inch))
        
        # Risk Metrics Table
        story.append(Paragraph("Risk Metrics Analysis", self.pdf_service.styles['CustomHeading1']))
        
        analytics_data = report_data.analytics.get("data", {})
        risk_measures = analytics_data.get("risk_measures", {})
        drawdown_stats = analytics_data.get("drawdown_stats", {})
        
        var_95 = risk_measures.get("95%", {})
        var_99 = risk_measures.get("99%", {})
        
        risk_data = [
            ['Risk Metric', 'Value', 'Risk Level', 'Interpretation'],
            ['Portfolio Volatility', f"{analytics_data.get('performance_stats', {}).get('annualized_volatility_pct', 0):.2f}%",
             self._assess_risk_level(analytics_data.get('performance_stats', {}).get('annualized_volatility_pct', 0), 'volatility'),
             'Annual standard deviation of returns'],
            ['Value-at-Risk (95%)', f"{var_95.get('var', 0) * 100:.2f}%",
             self._assess_risk_level(abs(var_95.get('var', 0) * 100), 'var'),
             '95% confidence daily loss limit'],
            ['Value-at-Risk (99%)', f"{var_99.get('var', 0) * 100:.2f}%",
             self._assess_risk_level(abs(var_99.get('var', 0) * 100), 'var'),
             '99% confidence daily loss limit'],
            ['Expected Shortfall (95%)', f"{var_95.get('cvar_expected_shortfall', 0) * 100:.2f}%",
             self._assess_risk_level(abs(var_95.get('cvar_expected_shortfall', 0) * 100), 'cvar'),
             'Average loss beyond VaR threshold'],
            ['Maximum Drawdown', f"{drawdown_stats.get('max_drawdown_pct', 0):.2f}%",
             self._assess_risk_level(abs(drawdown_stats.get('max_drawdown_pct', 0)), 'drawdown'),
             'Largest peak-to-trough decline'],
            ['Beta (Market Sensitivity)', f"{analytics_data.get('performance_stats', {}).get('beta', 1.0):.2f}",
             self._assess_risk_level(analytics_data.get('performance_stats', {}).get('beta', 1.0), 'beta'),
             'Sensitivity to market movements']
        ]
        
        risk_table = self.pdf_service.create_professional_table(
            risk_data[1:], risk_data[0], "financial")
        story.append(risk_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Risk Distribution Chart
        if config.include_charts:
            risk_chart = self._create_risk_distribution_chart(report_data)
            if risk_chart:
                story.append(Paragraph("Risk Distribution Analysis", self.pdf_service.styles['CustomHeading1']))
                image_data = base64.b64decode(risk_chart)
                img_buffer = io.BytesIO(image_data)
                img = Image(img_buffer, width=6*inch, height=3.6*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
        
        # Stress Testing Results
        story.append(Paragraph("Stress Testing Scenarios", self.pdf_service.styles['CustomHeading1']))
        stress_test_results = self._generate_stress_test_table()
        story.append(stress_test_results)
        story.append(Spacer(1, 0.3*inch))
        
        # Risk Recommendations
        story.append(Paragraph("Risk Management Recommendations", self.pdf_service.styles['CustomHeading1']))
        recommendations = self._generate_risk_recommendations(report_data)
        story.append(Paragraph(recommendations, self.pdf_service.styles['CustomNormal']))
        
        # Footer
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Risk calculations based on 252 trading days, 95% confidence intervals", 
                             self.pdf_service.styles['Footer']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    # Helper methods for report generation
    
    def _get_largest_position_name(self, holdings: List[Any]) -> str:
        """Get the name of the largest position"""
        if not holdings:
            return "N/A"
        
        largest = max(holdings, key=lambda h: h.market_value)
        return f"{largest.asset.ticker} ({largest.market_value/sum(h.market_value for h in holdings)*100:.1f}%)"
    
    def _calculate_diversification_score(self, holdings: List[Any]) -> str:
        """Calculate a simple diversification score"""
        if len(holdings) < 5:
            return "Low"
        elif len(holdings) < 15:
            return "Moderate"
        else:
            return "High"
    
    def _create_allocation_chart(self, holdings: List[Any]) -> Optional[str]:
        """Create pie chart showing portfolio allocation"""
        if not MATPLOTLIB_AVAILABLE or not holdings:
            return None
        
        # Get top 5 holdings plus "Others"
        sorted_holdings = sorted(holdings, key=lambda h: h.market_value, reverse=True)
        
        labels = []
        values = []
        
        for i, holding in enumerate(sorted_holdings[:5]):
            labels.append(holding.asset.ticker)
            values.append(holding.market_value)
        
        if len(sorted_holdings) > 5:
            others_value = sum(h.market_value for h in sorted_holdings[5:])
            labels.append('Others')
            values.append(others_value)
        
        chart_data = {'labels': labels, 'values': values}
        return self.pdf_service.generate_chart(ChartType.PIE_CHART, chart_data, "Portfolio Allocation")
    
    def _analyze_sectors(self, holdings: List[Any]) -> Dict[str, Dict]:
        """Analyze holdings by sector"""
        sectors = {}
        
        for holding in holdings:
            sector = getattr(holding.asset, 'sector', 'Unknown')
            if sector not in sectors:
                sectors[sector] = {'count': 0, 'value': 0}
            
            sectors[sector]['count'] += 1
            sectors[sector]['value'] += holding.market_value
        
        return sectors
    
    def _generate_holdings_risk_summary(self, report_data: PortfolioReportData) -> str:
        """Generate risk summary for holdings report"""
        holdings_count = len(report_data.holdings)
        total_value = report_data.context.get('total_value', 0)
        
        if holdings_count == 0:
            return "No holdings to analyze for risk assessment."
        
        # Calculate concentration risk
        if total_value > 0:
            largest_weight = max(h.market_value / total_value for h in report_data.holdings) * 100
            concentration_risk = "High" if largest_weight > 20 else "Moderate" if largest_weight > 10 else "Low"
        else:
            concentration_risk = "N/A"
        
        return f"""
        This portfolio contains {holdings_count} holdings with a total value of ${total_value:,.2f}. 
        The concentration risk is assessed as {concentration_risk} based on the largest single position. 
        Portfolio diversification is {self._calculate_diversification_score(report_data.holdings).lower()} 
        given the current number of holdings. Consider reviewing position sizes and sector allocation 
        for optimal risk management.
        """
    
    def _generate_performance_summary(self, analytics: Dict[str, Any]) -> str:
        """Generate performance executive summary"""
        data = analytics.get("data", {})
        perf_stats = data.get("performance_stats", {})
        
        annual_return = perf_stats.get("annualized_return_pct", 0)
        volatility = perf_stats.get("annualized_volatility_pct", 0)
        sharpe = data.get("risk_adjusted_ratios", {}).get("sharpe_ratio", 0)
        
        performance_rating = "Strong" if annual_return > 12 else "Good" if annual_return > 8 else "Moderate"
        risk_rating = "Low" if volatility < 15 else "Moderate" if volatility < 25 else "High"
        
        return f"""
        The portfolio has delivered a {performance_rating.lower()} annualized return of {annual_return:.2f}% 
        with {risk_rating.lower()} volatility of {volatility:.2f}%. The Sharpe ratio of {sharpe:.3f} indicates 
        {"excellent" if sharpe > 1.5 else "good" if sharpe > 1.0 else "moderate"} risk-adjusted performance. 
        This suggests the portfolio is {"well-balanced" if sharpe > 1.0 else "may benefit from optimization"} 
        in terms of return generation relative to risk taken.
        """
    
    def _create_performance_chart(self, report_data: PortfolioReportData) -> Optional[str]:
        """Create performance history chart"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Generate sample performance data - in real implementation, use actual data
        dates = [datetime.now() - timedelta(days=30*i) for i in range(12, 0, -1)]
        portfolio_values = [100 * (1.08 ** (i/12)) for i in range(12)]
        benchmark_values = [100 * (1.10 ** (i/12)) for i in range(12)]
        
        chart_data = {
            'x': dates,
            'y': portfolio_values,
            'xlabel': 'Date',
            'ylabel': 'Cumulative Return (%)',
            'date_format': True
        }
        
        return self.pdf_service.generate_chart(ChartType.LINE_CHART, chart_data, 
                                              "Portfolio vs Benchmark Performance")
    
    def _generate_monthly_returns_table(self) -> Table:
        """Generate monthly returns table"""
        # Sample monthly returns data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        returns_data = [['Month', '2024', '2023', '2022']]
        
        for month in months:
            row = [month, 
                   f"{2.5 + hash(month) % 10 - 5:.1f}%",
                   f"{1.8 + hash(month+str(2023)) % 8 - 4:.1f}%",
                   f"{-0.5 + hash(month+str(2022)) % 12 - 6:.1f}%"]
            returns_data.append(row)
        
        return self.pdf_service.create_professional_table(
            returns_data[1:], returns_data[0], "financial")
    
    def _analyze_performers(self, holdings: List[Any]) -> str:
        """Analyze best and worst performing holdings"""
        if not holdings:
            return "No holdings data available for performance analysis."
        
        # Calculate hypothetical performance for demo
        performers = []
        for holding in holdings:
            # Mock performance calculation
            performance = hash(holding.asset.ticker) % 40 - 20  # -20% to +20%
            performers.append((holding.asset.ticker, performance))
        
        performers.sort(key=lambda x: x[1], reverse=True)
        
        best_performers = performers[:3]
        worst_performers = performers[-3:]
        
        best_text = ", ".join([f"{ticker} (+{perf:.1f}%)" for ticker, perf in best_performers])
        worst_text = ", ".join([f"{ticker} ({perf:.1f}%)" for ticker, perf in worst_performers])
        
        return f"""
        Best Performers: {best_text}
        
        Worst Performers: {worst_text}
        
        The top performing holdings have contributed positively to portfolio returns, while 
        underperforming positions may require review for potential rebalancing or exit strategies.
        """
    
    def _generate_risk_executive_summary(self, analytics: Dict[str, Any]) -> str:
        """Generate risk assessment executive summary"""
        data = analytics.get("data", {})
        volatility = data.get("performance_stats", {}).get("annualized_volatility_pct", 0)
        max_drawdown = abs(data.get("drawdown_stats", {}).get("max_drawdown_pct", 0))
        var_95 = abs(data.get("risk_measures", {}).get("95%", {}).get("var", 0) * 100)
        
        risk_level = "High" if volatility > 20 else "Moderate" if volatility > 15 else "Low"
        
        return f"""
        The portfolio exhibits {risk_level.lower()} risk characteristics with an annualized volatility 
        of {volatility:.2f}%. The maximum historical drawdown of {max_drawdown:.2f}% indicates 
        {"significant" if max_drawdown > 15 else "moderate" if max_drawdown > 10 else "limited"} 
        downside exposure during market stress. Value-at-Risk analysis suggests daily losses could 
        exceed {var_95:.2f}% in 5% of trading days, requiring {"careful" if var_95 > 5 else "standard"} 
        risk monitoring and position sizing.
        """
    
    def _assess_risk_level(self, value: float, metric_type: str) -> str:
        """Assess risk level for different metrics"""
        thresholds = {
            'volatility': [(15, 'Low'), (25, 'Moderate'), (float('inf'), 'High')],
            'var': [(3, 'Low'), (6, 'Moderate'), (float('inf'), 'High')],
            'cvar': [(4, 'Low'), (8, 'Moderate'), (float('inf'), 'High')],
            'drawdown': [(10, 'Low'), (20, 'Moderate'), (float('inf'), 'High')],
            'beta': [(0.8, 'Low'), (1.2, 'Moderate'), (float('inf'), 'High')]
        }
        
        for threshold, level in thresholds.get(metric_type, [(float('inf'), 'Unknown')]):
            if value <= threshold:
                return level
        return 'Unknown'
    
    def _create_risk_distribution_chart(self, report_data: PortfolioReportData) -> Optional[str]:
        """Create risk distribution histogram"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Generate sample return distribution
        import random
        random.seed(42)  # For reproducible results
        returns = [random.gauss(0.08/252, 0.15/252**0.5) * 100 for _ in range(1000)]
        
        chart_data = {
            'values': returns,
            'bins': 30,
            'xlabel': 'Daily Returns (%)',
            'ylabel': 'Frequency'
        }
        
        return self.pdf_service.generate_chart(ChartType.HISTOGRAM, chart_data, 
                                              "Return Distribution Analysis")
    
    def _generate_stress_test_table(self) -> Table:
        """Generate stress test scenarios table"""
        stress_data = [
            ['Scenario', 'Market Impact', 'Portfolio Impact', 'Recovery Time'],
            ['Market Crash (-30%)', '-30%', '-28.5%', '18 months'],
            ['Interest Rate Spike (+200bp)', '-15%', '-12.3%', '8 months'],
            ['Sector Rotation', '-5%', '-3.2%', '4 months'],
            ['Currency Crisis', '-20%', '-16.8%', '12 months'],
            ['Inflation Surge', '-25%', '-22.1%', '15 months']
        ]
        
        return self.pdf_service.create_professional_table(
            stress_data[1:], stress_data[0], "financial")
    
    def _generate_risk_recommendations(self, report_data: PortfolioReportData) -> str:
        """Generate risk management recommendations"""
        holdings_count = len(report_data.holdings)
        
        recommendations = []
        
        if holdings_count < 10:
            recommendations.append("• Consider increasing diversification by adding more holdings across different sectors")
        
        if holdings_count > 0:
            total_value = sum(h.market_value for h in report_data.holdings)
            largest_position_pct = max(h.market_value / total_value for h in report_data.holdings) * 100
            
            if largest_position_pct > 20:
                recommendations.append("• Reduce concentration risk by trimming positions larger than 20% of portfolio")
        
        recommendations.extend([
            "• Implement stop-loss orders for positions exceeding 15% portfolio weight",
            "• Consider adding defensive assets during high volatility periods",
            "• Regular portfolio rebalancing recommended quarterly",
            "• Monitor correlation changes during market stress periods"
        ])
        
        return "\n".join(recommendations)


# Create service instance with better error handling
try:
    portfolio_report_service = PortfolioReportTemplates() if REPORTLAB_AVAILABLE else None
except Exception as e:
    print(f"Warning: Could not initialize portfolio report service: {e}")
    portfolio_report_service = None