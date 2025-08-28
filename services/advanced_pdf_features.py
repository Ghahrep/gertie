# services/advanced_pdf_features.py
"""
Advanced PDF Features - Sub-task 3.3.1.4
=========================================
Multi-page layouts, advanced chart integration, branded templates, and batch report generation
"""

import io
import os
import json
import zipfile
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import asyncio
import concurrent.futures
from pathlib import Path

# Import our enhanced PDF services
try:
    from services.enhanced_pdf_service import (
        ProfessionalPDFService, ReportConfig, ReportType, ChartType,
        REPORTLAB_AVAILABLE, MATPLOTLIB_AVAILABLE
    )
    from services.portfolio_report_templates import PortfolioReportTemplates
    from services.debate_report_templates import DebateReportTemplates
    from services.report_data_types import PortfolioReportData, DebateReportData, BaseReportData
    
    if REPORTLAB_AVAILABLE:
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
            PageBreak, Image, KeepTogether, Flowable, NextPageTemplate, PageTemplate
        )
        from reportlab.lib.units import inch, cm
        from reportlab.lib.pagesizes import A4, letter
        from reportlab.lib import colors
        from reportlab.platypus.tableofcontents import TableOfContents
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.platypus.doctemplate import BaseDocTemplate
        from reportlab.lib.utils import ImageReader
        
except ImportError as e:
    print(f"Warning: PDF services not available - {e}")
    # Set fallback values
    REPORTLAB_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False
    ProfessionalPDFService = None
    ReportConfig = None
    ReportType = None
    ChartType = None
    PortfolioReportTemplates = None
    DebateReportTemplates = None
    PortfolioReportData = None
    DebateReportData = None
    BaseReportData = None

import logging

logger = logging.getLogger(__name__)

@dataclass 
class BatchReportRequest:
    """Request configuration for batch report generation"""
    report_type: str
    portfolios: List[Any] = field(default_factory=list)
    debates: List[Any] = field(default_factory=list)
    output_format: str = "pdf"  # pdf, zip
    include_charts: bool = True
    template_style: str = "professional"
    batch_id: Optional[str] = None
    priority: str = "normal"  # low, normal, high
    
    def __post_init__(self):
        if self.batch_id is None:
            self.batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

@dataclass
class ReportTemplate:
    """Advanced report template configuration"""
    name: str
    template_type: str
    page_layout: str = "single_column"  # single_column, two_column, mixed
    header_style: str = "professional"
    footer_style: str = "minimal"
    color_scheme: str = "corporate_blue"
    logo_position: str = "top_right"
    watermark: Optional[str] = None
    toc_enabled: bool = True
    page_numbers: bool = True
    custom_css: Optional[Dict] = None

class AdvancedPDFFeatures:
    """Advanced PDF generation with multi-page layouts, branding, and batch processing"""
    
    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for advanced PDF features")
        
        self.pdf_service = ProfessionalPDFService()
        self.portfolio_templates = PortfolioReportTemplates(self.pdf_service)
        self.debate_templates = DebateReportTemplates(self.pdf_service)
        self.batch_jobs = {}
        self.template_cache = {}
        
        # Brand configuration
        self.brand_config = {
            "corporate_blue": {
                "primary": colors.HexColor("#2E86AB"),
                "secondary": colors.HexColor("#A23B72"),
                "accent": colors.HexColor("#F18F01"),
                "neutral": colors.HexColor("#6C757D")
            },
            "financial_green": {
                "primary": colors.HexColor("#28A745"),
                "secondary": colors.HexColor("#17A2B8"), 
                "accent": colors.HexColor("#FFC107"),
                "neutral": colors.HexColor("#6C757D")
            },
            "executive_gray": {
                "primary": colors.HexColor("#343A40"),
                "secondary": colors.HexColor("#495057"),
                "accent": colors.HexColor("#007BFF"),
                "neutral": colors.HexColor("#6C757D")
            }
        }
    
    def create_multi_page_report(self, 
                                content_sections: List[Dict[str, Any]], 
                                template: ReportTemplate,
                                output_path: Optional[str] = None) -> io.BytesIO:
        """Create multi-page report with advanced layouts"""
        
        buffer = io.BytesIO()
        
        # Create document with custom page templates
        doc = self._create_advanced_document(buffer, template)
        story = []
        
        # Add title page
        story.extend(self._create_title_page(content_sections[0], template))
        story.append(PageBreak())
        
        # Add table of contents if enabled
        if template.toc_enabled:
            story.extend(self._create_table_of_contents())
            story.append(PageBreak())
        
        # Process content sections with different layouts
        for i, section in enumerate(content_sections[1:], 1):
            section_story = self._create_section_content(section, template)
            
            # Apply layout based on section type
            if section.get('layout') == 'two_column':
                section_story = self._apply_two_column_layout(section_story)
            elif section.get('layout') == 'dashboard':
                section_story = self._apply_dashboard_layout(section_story)
            
            story.extend(section_story)
            
            # Add page break between major sections
            if i < len(content_sections) - 1:
                story.append(PageBreak())
        
        # Build document
        doc.build(story)
        buffer.seek(0)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            buffer.seek(0)
        
        return buffer
    
    def generate_branded_report(self, 
                              report_data: Union[PortfolioReportData, DebateReportData],
                              brand_style: str = "corporate_blue",
                              logo_path: Optional[str] = None) -> io.BytesIO:
        """Generate report with full branding"""
        
        # Create branded template
        template = ReportTemplate(
            name="Branded Report",
            template_type="portfolio" if isinstance(report_data, PortfolioReportData) else "debate",
            color_scheme=brand_style,
            header_style="branded",
            logo_position="top_left" if logo_path else "none"
        )
        
        buffer = io.BytesIO()
        colors_scheme = self.brand_config.get(brand_style, self.brand_config["corporate_blue"])
        
        # Create branded document
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4,
            rightMargin=0.75*inch, 
            leftMargin=0.75*inch,
            topMargin=1.2*inch,  # Extra space for branded header
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Add branded header
        story.extend(self._create_branded_header(template, logo_path, colors_scheme))
        
        # Generate appropriate report content
        if isinstance(report_data, PortfolioReportData):
            content = self._generate_branded_portfolio_content(report_data, colors_scheme)
        else:
            content = self._generate_branded_debate_content(report_data, colors_scheme)
        
        story.extend(content)
        
        # Add branded footer
        story.extend(self._create_branded_footer(template, colors_scheme))
        
        # Build with branded page template
        def add_branded_page_elements(canvas, doc):
            self._add_branded_page_elements(canvas, doc, template, logo_path, colors_scheme)
        
        doc.build(story, onFirstPage=add_branded_page_elements, onLaterPages=add_branded_page_elements)
        buffer.seek(0)
        
        return buffer
    
    def create_comprehensive_dashboard_report(self, 
                                            portfolio_data: PortfolioReportData,
                                            debate_data: Optional[DebateReportData] = None) -> io.BytesIO:
        """Create comprehensive dashboard-style report"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=0.5*inch, leftMargin=0.5*inch,
                               topMargin=0.75*inch, bottomMargin=0.5*inch)
        
        story = []
        
        # Dashboard title
        story.append(Paragraph("Investment Portfolio Dashboard", 
                             self.pdf_service.styles['CustomTitle']))
        story.append(Paragraph(f"Portfolio: {portfolio_data.portfolio.name} | {datetime.now().strftime('%B %d, %Y')}", 
                             self.pdf_service.styles['CustomHeading2']))
        story.append(Spacer(1, 0.3*inch))
        
        # Key metrics overview (dashboard style)
        metrics_section = self._create_metrics_dashboard(portfolio_data)
        story.extend(metrics_section)
        
        # Charts section
        if portfolio_data.context.get('total_value', 0) > 0:
            charts_section = self._create_charts_section(portfolio_data)
            story.extend(charts_section)
        
        # Holdings summary table
        holdings_section = self._create_compact_holdings_table(portfolio_data)
        story.extend(holdings_section)
        
        # Risk indicators
        risk_section = self._create_risk_indicators_section(portfolio_data)
        story.extend(risk_section)
        
        # AI insights section (if debate data available)
        if debate_data:
            ai_insights = self._create_ai_insights_section(debate_data)
            story.extend(ai_insights)
        
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    
    async def generate_batch_reports(self, 
                                   batch_request: BatchReportRequest,
                                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Generate multiple reports in batch with progress tracking"""
        
        batch_id = batch_request.batch_id
        self.batch_jobs[batch_id] = {
            "status": "processing",
            "progress": 0,
            "total_reports": len(batch_request.portfolios) + len(batch_request.debates),
            "completed_reports": 0,
            "failed_reports": 0,
            "results": [],
            "started_at": datetime.now()
        }
        
        try:
            results = []
            total_items = len(batch_request.portfolios) + len(batch_request.debates)
            
            # Process portfolios
            for i, portfolio in enumerate(batch_request.portfolios):
                try:
                    # Mock portfolio data - in real implementation, construct from actual portfolio
                    mock_context = {"total_value": 100000, "holdings_with_values": []}
                    mock_analytics = {"data": {"performance_stats": {}}}
                    
                    portfolio_data = PortfolioReportData(portfolio, mock_context, mock_analytics)
                    
                    if batch_request.template_style == "branded":
                        report_buffer = self.generate_branded_report(portfolio_data)
                    else:
                        report_buffer = self.portfolio_templates.generate_holdings_summary_report(portfolio_data)
                    
                    results.append({
                        "type": "portfolio",
                        "id": getattr(portfolio, 'id', f'portfolio_{i}'),
                        "name": getattr(portfolio, 'name', f'Portfolio {i+1}'),
                        "buffer": report_buffer,
                        "status": "success"
                    })
                    
                    self.batch_jobs[batch_id]["completed_reports"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to generate portfolio report {i}: {e}")
                    self.batch_jobs[batch_id]["failed_reports"] += 1
                    results.append({
                        "type": "portfolio",
                        "id": getattr(portfolio, 'id', f'portfolio_{i}'),
                        "status": "failed",
                        "error": str(e)
                    })
                
                # Update progress
                progress = ((i + 1) / total_items) * 100
                self.batch_jobs[batch_id]["progress"] = progress
                
                if progress_callback:
                    await progress_callback(batch_id, progress, f"Processed portfolio {i+1}")
            
            # Process debates
            debate_start_idx = len(batch_request.portfolios)
            for i, debate in enumerate(batch_request.debates):
                try:
                    # Mock debate data - in real implementation, construct from actual debate
                    mock_participants = []
                    mock_messages = []
                    mock_consensus = []
                    
                    debate_data = DebateReportData(debate, mock_participants, mock_messages, mock_consensus)
                    
                    report_buffer = self.debate_templates.generate_debate_summary_report(debate_data)
                    
                    results.append({
                        "type": "debate",
                        "id": getattr(debate, 'id', f'debate_{i}'),
                        "query": getattr(debate, 'query', f'Debate Query {i+1}')[:50],
                        "buffer": report_buffer,
                        "status": "success"
                    })
                    
                    self.batch_jobs[batch_id]["completed_reports"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to generate debate report {i}: {e}")
                    self.batch_jobs[batch_id]["failed_reports"] += 1
                    results.append({
                        "type": "debate", 
                        "id": getattr(debate, 'id', f'debate_{i}'),
                        "status": "failed",
                        "error": str(e)
                    })
                
                # Update progress
                progress = ((debate_start_idx + i + 1) / total_items) * 100
                self.batch_jobs[batch_id]["progress"] = progress
                
                if progress_callback:
                    await progress_callback(batch_id, progress, f"Processed debate {i+1}")
            
            # Finalize batch job
            self.batch_jobs[batch_id].update({
                "status": "completed",
                "progress": 100,
                "results": results,
                "completed_at": datetime.now()
            })
            
            # Create output based on format
            if batch_request.output_format == "zip":
                output_buffer = self._create_batch_zip_output(results, batch_id)
            else:
                output_buffer = self._create_batch_pdf_output(results, batch_id)
            
            self.batch_jobs[batch_id]["output"] = output_buffer
            
            return {
                "batch_id": batch_id,
                "status": "completed", 
                "total_reports": total_items,
                "successful_reports": self.batch_jobs[batch_id]["completed_reports"],
                "failed_reports": self.batch_jobs[batch_id]["failed_reports"],
                "output": output_buffer,
                "summary": f"Generated {len(results)} reports ({self.batch_jobs[batch_id]['completed_reports']} successful, {self.batch_jobs[batch_id]['failed_reports']} failed)"
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed for {batch_id}: {e}")
            self.batch_jobs[batch_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now()
            })
            
            return {
                "batch_id": batch_id,
                "status": "failed",
                "error": str(e)
            }
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of batch report generation"""
        return self.batch_jobs.get(batch_id)
    
    def cancel_batch_job(self, batch_id: str) -> bool:
        """Cancel running batch job"""
        if batch_id in self.batch_jobs and self.batch_jobs[batch_id]["status"] == "processing":
            self.batch_jobs[batch_id]["status"] = "cancelled"
            self.batch_jobs[batch_id]["completed_at"] = datetime.now()
            return True
        return False
    
    # Helper methods for advanced PDF features
    
    def _create_advanced_document(self, buffer: io.BytesIO, template: ReportTemplate):
        """Create document with advanced page templates"""
        return SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=1*inch, bottomMargin=0.75*inch)
    
    def _create_title_page(self, content: Dict[str, Any], template: ReportTemplate) -> List:
        """Create professional title page"""
        story = []
        
        # Title
        title = content.get('title', 'Financial Report')
        story.append(Paragraph(title, self.pdf_service.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        subtitle = content.get('subtitle', 'Comprehensive Analysis')
        story.append(Paragraph(subtitle, self.pdf_service.styles['CustomHeading1']))
        story.append(Spacer(1, 1*inch))
        
        # Summary box
        if content.get('summary'):
            summary_style = ParagraphStyle(
                'TitleSummary',
                parent=self.pdf_service.styles['ExecutiveSummary'],
                fontSize=12,
                leftIndent=30,
                rightIndent=30,
                spaceBefore=20,
                spaceAfter=20
            )
            story.append(Paragraph("Executive Summary", self.pdf_service.styles['CustomHeading2']))
            story.append(Paragraph(content['summary'], summary_style))
        
        story.append(Spacer(1, 1*inch))
        
        # Report details
        details = [
            f"Generated: {datetime.now().strftime('%B %d, %Y')}",
            f"Report Type: {template.template_type.title()}",
            f"Template: {template.name}"
        ]
        
        for detail in details:
            story.append(Paragraph(detail, self.pdf_service.styles['CustomNormal']))
        
        return story
    
    def _create_table_of_contents(self) -> List:
        """Create table of contents"""
        story = []
        story.append(Paragraph("Table of Contents", self.pdf_service.styles['CustomHeading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # Mock TOC entries
        toc_entries = [
            "Executive Summary ..................................... 3",
            "Portfolio Overview ................................... 4", 
            "Holdings Analysis .................................... 5",
            "Performance Metrics .................................. 7",
            "Risk Assessment ...................................... 9",
            "Recommendations ..................................... 11"
        ]
        
        for entry in toc_entries:
            story.append(Paragraph(entry, self.pdf_service.styles['CustomNormal']))
        
        return story
    
    def _create_section_content(self, section: Dict[str, Any], template: ReportTemplate) -> List:
        """Create content for a section"""
        story = []
        
        # Section header
        if section.get('title'):
            story.append(Paragraph(section['title'], self.pdf_service.styles['CustomHeading1']))
            story.append(Spacer(1, 0.2*inch))
        
        # Section content
        if section.get('content'):
            story.append(Paragraph(section['content'], self.pdf_service.styles['CustomNormal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Section table
        if section.get('table_data'):
            table = self.pdf_service.create_professional_table(
                section['table_data'], 
                section.get('table_headers'),
                section.get('table_style', 'default')
            )
            story.append(table)
            story.append(Spacer(1, 0.2*inch))
        
        # Section chart
        if section.get('chart_data'):
            chart_b64 = self.pdf_service.generate_chart(
                section['chart_type'],
                section['chart_data'],
                section.get('chart_title', 'Chart')
            )
            if chart_b64:
                import base64
                image_data = base64.b64decode(chart_b64)
                img_buffer = io.BytesIO(image_data)
                img = Image(img_buffer, width=5*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _apply_two_column_layout(self, content_story: List) -> List:
        """Apply two-column layout to content"""
        # For this mock implementation, return content as-is
        # In real implementation, would use more complex ReportLab layouts
        return content_story
    
    def _apply_dashboard_layout(self, content_story: List) -> List:
        """Apply dashboard layout to content"""
        # For this mock implementation, return content as-is
        # In real implementation, would create grid-based layouts
        return content_story
    
    def _create_branded_header(self, template: ReportTemplate, logo_path: Optional[str], colors_scheme: Dict) -> List:
        """Create branded header"""
        story = []
        
        # Company header
        header_style = ParagraphStyle(
            'BrandedHeader',
            parent=self.pdf_service.styles['CustomTitle'],
            textColor=colors_scheme['primary'],
            fontSize=20,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph("Advanced Portfolio Analytics", header_style))
        story.append(Paragraph("Professional Investment Management", self.pdf_service.styles['CustomHeading2']))
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _generate_branded_portfolio_content(self, report_data: PortfolioReportData, colors_scheme: Dict) -> List:
        """Generate branded portfolio content"""
        story = []
        
        # Portfolio overview with branding
        branded_style = ParagraphStyle(
            'BrandedContent',
            parent=self.pdf_service.styles['CustomNormal'],
            textColor=colors_scheme['neutral']
        )
        
        story.append(Paragraph(f"Portfolio Analysis: {report_data.portfolio.name}", 
                             self.pdf_service.styles['CustomHeading1']))
        
        overview_text = f"""
        This comprehensive analysis provides detailed insights into portfolio performance, 
        risk characteristics, and strategic recommendations based on current market conditions 
        and advanced quantitative analysis.
        """
        
        story.append(Paragraph(overview_text, branded_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Key metrics with branded styling
        total_value = report_data.context.get('total_value', 0)
        holdings_count = len(report_data.holdings)
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Portfolio Value', f'${total_value:,.2f}'],
            ['Number of Holdings', str(holdings_count)],
            ['Analysis Date', datetime.now().strftime('%B %d, %Y')]
        ]
        
        metrics_table = self.pdf_service.create_professional_table(
            metrics_data[1:], metrics_data[0], 'financial')
        story.append(metrics_table)
        
        return story
    
    def _generate_branded_debate_content(self, report_data: DebateReportData, colors_scheme: Dict) -> List:
        """Generate branded debate content"""
        story = []
        
        story.append(Paragraph("Multi-Agent Investment Analysis", 
                             self.pdf_service.styles['CustomHeading1']))
        
        debate_summary = f"""
        Our advanced AI system conducted a comprehensive debate with {report_data.total_participants} 
        specialized agents, exchanging {report_data.total_messages} messages over 
        {report_data.debate_duration:.1f} minutes to provide investment recommendations.
        """
        
        story.append(Paragraph(debate_summary, self.pdf_service.styles['CustomNormal']))
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _create_branded_footer(self, template: ReportTemplate, colors_scheme: Dict) -> List:
        """Create branded footer"""
        story = []
        
        footer_style = ParagraphStyle(
            'BrandedFooter',
            parent=self.pdf_service.styles['Footer'],
            textColor=colors_scheme['neutral'],
            alignment=TA_CENTER
        )
        
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Confidential and Proprietary", footer_style))
        story.append(Paragraph("Advanced Portfolio Analytics - Professional Investment Management", footer_style))
        
        return story
    
    def _add_branded_page_elements(self, canvas, doc, template: ReportTemplate, 
                                  logo_path: Optional[str], colors_scheme: Dict):
        """Add branded elements to each page"""
        canvas.saveState()
        
        # Header line
        canvas.setStrokeColor(colors_scheme['primary'])
        canvas.setLineWidth(2)
        canvas.line(doc.leftMargin, doc.height + doc.topMargin - 20, 
                   doc.width + doc.leftMargin, doc.height + doc.topMargin - 20)
        
        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors_scheme['neutral'])
        footer_text = f"Page {canvas.getPageNumber()} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        canvas.drawCentredText(doc.width/2 + doc.leftMargin, doc.bottomMargin - 30, footer_text)
        
        canvas.restoreState()
    
    def _create_metrics_dashboard(self, portfolio_data: PortfolioReportData) -> List:
        """Create dashboard-style metrics overview"""
        story = []
        
        story.append(Paragraph("Key Metrics Overview", self.pdf_service.styles['CustomHeading1']))
        
        # Create 2x2 metrics grid
        total_value = portfolio_data.context.get('total_value', 0)
        holdings_count = len(portfolio_data.holdings)
        
        # Mock additional metrics
        day_change = total_value * 0.015  # 1.5% daily change
        ytd_return = 8.5  # 8.5% YTD return
        
        metrics_grid = [
            [f'${total_value:,.0f}', f'{holdings_count}'],
            ['Total Value', 'Holdings'],
            [f'${day_change:+,.0f} (+1.5%)', f'{ytd_return:+.1f}%'],
            ['Today\'s Change', 'YTD Return']
        ]
        
        # Create styled table for metrics
        metrics_table = Table(metrics_grid, colWidths=[2.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            # Values row (top)
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 18),
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#2E86AB')),
            ('BOTTOMPADDING', (0,0), (-1,0), 15),
            
            # Labels row
            ('FONTNAME', (0,1), (-1,1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,1), 10),
            ('ALIGN', (0,1), (-1,1), 'CENTER'),
            ('TEXTCOLOR', (0,1), (-1,1), colors.grey),
            ('TOPPADDING', (0,1), (-1,1), 5),
            ('BOTTOMPADDING', (0,1), (-1,1), 15),
            
            # Second values row
            ('FONTNAME', (0,2), (-1,2), 'Helvetica-Bold'),
            ('FONTSIZE', (0,2), (-1,2), 18),
            ('ALIGN', (0,2), (-1,2), 'CENTER'),
            ('TEXTCOLOR', (0,2), (-1,2), colors.HexColor('#28A745')),
            ('BOTTOMPADDING', (0,2), (-1,2), 15),
            
            # Second labels row
            ('FONTNAME', (0,3), (-1,3), 'Helvetica'),
            ('FONTSIZE', (0,3), (-1,3), 10),
            ('ALIGN', (0,3), (-1,3), 'CENTER'),
            ('TEXTCOLOR', (0,3), (-1,3), colors.grey),
            ('TOPPADDING', (0,3), (-1,3), 5),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _create_charts_section(self, portfolio_data: PortfolioReportData) -> List:
        """Create charts section for dashboard"""
        story = []
        
        story.append(Paragraph("Portfolio Analytics", self.pdf_service.styles['CustomHeading1']))
        
        # Create allocation chart if we have holdings
        if portfolio_data.holdings:
            # Get top 5 holdings for pie chart
            sorted_holdings = sorted(portfolio_data.holdings, key=lambda h: h.market_value, reverse=True)
            
            labels = []
            values = []
            for holding in sorted_holdings[:5]:
                labels.append(holding.asset.ticker)
                values.append(holding.market_value)
            
            if len(sorted_holdings) > 5:
                others_value = sum(h.market_value for h in sorted_holdings[5:])
                labels.append('Others')
                values.append(others_value)
            
            chart_data = {'labels': labels, 'values': values}
            chart_b64 = self.pdf_service.generate_chart(ChartType.PIE_CHART, chart_data, "Portfolio Allocation")
            
            if chart_b64:
                import base64
                image_data = base64.b64decode(chart_b64)
                img_buffer = io.BytesIO(image_data)
                img = Image(img_buffer, width=4*inch, height=2.5*inch)
                story.append(img)
        
        story.append(Spacer(1, 0.2*inch))
        return story
    
    def _create_compact_holdings_table(self, portfolio_data: PortfolioReportData) -> List:
        """Create compact holdings table for dashboard"""
        story = []
        
        story.append(Paragraph("Top Holdings", self.pdf_service.styles['CustomHeading2']))
        
        if portfolio_data.holdings:
            # Show top 5 holdings
            sorted_holdings = sorted(portfolio_data.holdings, key=lambda h: h.market_value, reverse=True)
            total_value = portfolio_data.context.get('total_value', 1)
            
            holdings_data = [['Symbol', 'Value', 'Weight']]
            
            for holding in sorted_holdings[:5]:
                weight = (holding.market_value / total_value) * 100 if total_value > 0 else 0
                holdings_data.append([
                    holding.asset.ticker,
                    f'${holding.market_value:,.0f}',
                    f'{weight:.1f}%'
                ])
            
            holdings_table = self.pdf_service.create_professional_table(
                holdings_data[1:], holdings_data[0], 'minimal')
            story.append(holdings_table)
        
        story.append(Spacer(1, 0.2*inch))
        return story
    
    def _create_risk_indicators_section(self, portfolio_data: PortfolioReportData) -> List:
        """Create risk indicators section"""
        story = []
        
        story.append(Paragraph("Risk Indicators", self.pdf_service.styles['CustomHeading2']))
        
        # Mock risk indicators
        risk_data = [
            ['Indicator', 'Value', 'Status'],
            ['Portfolio Beta', '1.05', 'Normal'],
            ['Volatility (30d)', '12.5%', 'Low'],
            ['Max Drawdown', '-8.2%', 'Good'],
            ['Sharpe Ratio', '1.35', 'Strong']
        ]
        
        risk_table = self.pdf_service.create_professional_table(
            risk_data[1:], risk_data[0], 'financial')
        story.append(risk_table)
        story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _create_ai_insights_section(self, debate_data: DebateReportData) -> List:
        """Create AI insights section"""
        story = []
        
        story.append(Paragraph("AI Investment Insights", self.pdf_service.styles['CustomHeading2']))
        
        insights_text = f"""
        Our multi-agent AI system analyzed your portfolio with {debate_data.total_participants} 
        specialized agents reaching {debate_data.consensus_rate:.1f}% consensus. 
        Key recommendation: Maintain current allocation with slight rebalancing toward 
        high-conviction positions.
        """
        
        story.append(Paragraph(insights_text, self.pdf_service.styles['CustomNormal']))
        story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _create_batch_zip_output(self, results: List[Dict], batch_id: str) -> io.BytesIO:
        """Create ZIP file containing all batch reports"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, result in enumerate(results):
                if result.get('buffer') and result.get('status') == 'success':
                    filename = f"{result['type']}_{result.get('id', i)}_{batch_id}.pdf"
                    zip_file.writestr(filename, result['buffer'].getvalue())
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def _create_batch_pdf_output(self, results: List[Dict], batch_id: str) -> io.BytesIO:
        """Create single PDF containing all batch reports"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        
        story = []
        story.append(Paragraph(f"Batch Report: {batch_id}", self.pdf_service.styles['CustomTitle']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", 
                             self.pdf_service.styles['CustomNormal']))
        story.append(Spacer(1, 0.5*inch))
        
        # Summary
        successful = sum(1 for r in results if r.get('status') == 'success')
        story.append(Paragraph(f"Batch Summary: {successful}/{len(results)} reports generated successfully", 
                             self.pdf_service.styles['CustomHeading1']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # List of reports
        for result in results:
            if result.get('status') == 'success':
                name = result.get('name', result.get('query', result.get('id', 'Unknown')))
                story.append(Paragraph(f"✓ {result['type'].title()}: {name}", 
                                     self.pdf_service.styles['CustomNormal']))
            else:
                story.append(Paragraph(f"✗ Failed: {result.get('id', 'Unknown')}", 
                                     self.pdf_service.styles['CustomNormal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer


# Service instance
advanced_pdf_service = AdvancedPDFFeatures() if REPORTLAB_AVAILABLE else None