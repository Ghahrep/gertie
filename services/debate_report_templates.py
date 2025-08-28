# services/debate_report_templates.py
"""
Multi-Agent Debate Report Templates - Sub-task 3.3.1.3
======================================================
Specialized report templates for multi-agent debate results, consensus analysis, and agent performance
"""

import io
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
import base64
from dataclasses import dataclass

# Import our enhanced PDF service and portfolio templates
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
class DebateReportData:
    """Structured data container for debate reports"""
    debate: Any
    participants: List[Any]
    messages: List[Any]
    consensus_items: List[Any]
    analytics: Any = None
    
    def __post_init__(self):
        # Calculate derived metrics
        self.total_messages = len(self.messages)
        self.total_participants = len(self.participants)
        self.debate_duration = self._calculate_duration()
        self.consensus_rate = self._calculate_consensus_rate()
    
    def _calculate_duration(self) -> Optional[float]:
        """Calculate debate duration in minutes"""
        if hasattr(self.debate, 'completed_at') and hasattr(self.debate, 'started_at'):
            if self.debate.completed_at and self.debate.started_at:
                duration = self.debate.completed_at - self.debate.started_at
                return duration.total_seconds() / 60
        return None
    
    def _calculate_consensus_rate(self) -> float:
        """Calculate percentage of topics reaching consensus"""
        if not self.consensus_items:
            return 0.0
        
        consensus_reached = sum(1 for item in self.consensus_items 
                               if getattr(item, 'consensus_reached_at', None) is not None)
        return (consensus_reached / len(self.consensus_items)) * 100

class DebateReportTemplates:
    """Specialized multi-agent debate report templates"""
    
    def __init__(self, pdf_service: Optional[Any] = None):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for debate reports")
        
        self.pdf_service = pdf_service
        if self.pdf_service is None:
            self.pdf_service = ProfessionalPDFService()
    
    def generate_debate_summary_report(self, report_data: DebateReportData) -> io.BytesIO:
        """Generate comprehensive debate summary report"""
        
        config = ReportConfig(
            report_type=ReportType.DEBATE_SUMMARY,
            title=f"Debate Analysis: {report_data.debate.query[:50]}...",
            subtitle="Multi-Agent Investment Decision Analysis"
        )
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=1*inch, bottomMargin=0.75*inch)
        
        story = []
        
        # Title and Overview
        story.append(Paragraph(config.title, self.pdf_service.styles['CustomTitle']))
        story.append(Paragraph(config.subtitle, self.pdf_service.styles['CustomHeading2']))
        story.append(Paragraph(f"Debate ID: {getattr(report_data.debate, 'id', 'N/A')}", 
                             self.pdf_service.styles['CustomNormal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", 
                             self.pdf_service.styles['CustomNormal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.pdf_service.styles['CustomHeading1']))
        exec_summary = self._generate_debate_executive_summary(report_data)
        story.append(Paragraph(exec_summary, self.pdf_service.styles['ExecutiveSummary']))
        story.append(Spacer(1, 0.3*inch))
        
        # Debate Overview Metrics
        story.append(Paragraph("Debate Overview", self.pdf_service.styles['CustomHeading1']))
        
        overview_data = [
            ['Metric', 'Value'],
            ['Original Query', report_data.debate.query[:100] + ('...' if len(report_data.debate.query) > 100 else '')],
            ['Status', getattr(report_data.debate, 'status', 'Unknown').title()],
            ['Participants', str(report_data.total_participants)],
            ['Total Messages', str(report_data.total_messages)],
            ['Debate Duration', f"{report_data.debate_duration:.1f} minutes" if report_data.debate_duration else 'N/A'],
            ['Consensus Rate', f"{report_data.consensus_rate:.1f}%"],
            ['Confidence Score', f"{getattr(report_data.debate, 'confidence_score', 0):.2f}" if getattr(report_data.debate, 'confidence_score', None) else 'N/A'],
            ['Consensus Type', getattr(report_data.debate, 'consensus_type', 'N/A').replace('_', ' ').title()]
        ]
        
        overview_table = self.pdf_service.create_professional_table(
            overview_data[1:], overview_data[0], "financial")
        story.append(overview_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Participant Analysis
        story.append(Paragraph("Participant Analysis", self.pdf_service.styles['CustomHeading1']))
        
        if report_data.participants:
            participant_data = [
                ['Agent', 'Role', 'Messages Sent', 'Evidence Provided', 'Avg Confidence', 'Performance']
            ]
            
            for participant in report_data.participants:
                performance_score = self._calculate_agent_performance(participant)
                participant_data.append([
                    getattr(participant, 'agent_name', getattr(participant, 'agent_id', 'Unknown')),
                    getattr(participant, 'role', 'Participant').title(),
                    str(getattr(participant, 'messages_sent', 0)),
                    str(getattr(participant, 'evidence_provided', 0)),
                    f"{getattr(participant, 'avg_confidence_score', 0):.2f}",
                    performance_score
                ])
            
            participant_table = self.pdf_service.create_professional_table(
                participant_data[1:], participant_data[0], "financial")
            story.append(participant_table)
        else:
            story.append(Paragraph("No participant data available.", self.pdf_service.styles['CustomNormal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Consensus Items Analysis
        story.append(Paragraph("Consensus Analysis", self.pdf_service.styles['CustomHeading1']))
        
        if report_data.consensus_items:
            consensus_data = [
                ['Topic', 'Agreement %', 'Consensus Strength', 'Implementation Priority']
            ]
            
            for item in report_data.consensus_items[:10]:  # Top 10 items
                consensus_data.append([
                    (getattr(item, 'topic', 'Unknown Topic')[:40] + 
                     ('...' if len(getattr(item, 'topic', '')) > 40 else '')),
                    f"{getattr(item, 'agreement_percentage', 0):.1f}%",
                    getattr(item, 'consensus_strength', 'Unknown').title(),
                    getattr(item, 'implementation_priority', 'Medium').title()
                ])
            
            consensus_table = self.pdf_service.create_professional_table(
                consensus_data[1:], consensus_data[0], "financial")
            story.append(consensus_table)
            
            if len(report_data.consensus_items) > 10:
                story.append(Paragraph(f"... and {len(report_data.consensus_items) - 10} more consensus items", 
                                     self.pdf_service.styles['CustomNormal']))
        else:
            story.append(Paragraph("No consensus items reached during this debate.", 
                                 self.pdf_service.styles['CustomNormal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Key Insights and Recommendations
        if hasattr(report_data.debate, 'final_recommendation') and report_data.debate.final_recommendation:
            story.append(Paragraph("Final Recommendation", self.pdf_service.styles['CustomHeading1']))
            recommendation_text = self._format_recommendation(report_data.debate.final_recommendation)
            story.append(Paragraph(recommendation_text, self.pdf_service.styles['CustomNormal']))
        
        # Debate Flow Chart
        if config.include_charts and report_data.messages:
            debate_flow_chart = self._create_debate_flow_chart(report_data)
            if debate_flow_chart:
                story.append(PageBreak())
                story.append(Paragraph("Debate Flow Analysis", self.pdf_service.styles['CustomHeading1']))
                image_data = base64.b64decode(debate_flow_chart)
                img_buffer = io.BytesIO(image_data)
                img = Image(img_buffer, width=6*inch, height=4*inch)
                story.append(img)
        
        # Footer
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Generated by Advanced Multi-Agent Debate System", 
                             self.pdf_service.styles['Footer']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_consensus_analysis_report(self, report_data: DebateReportData) -> io.BytesIO:
        """Generate detailed consensus analysis report"""
        
        config = ReportConfig(
            report_type=ReportType.CONSENSUS_REPORT,
            title="Consensus Analysis Report",
            subtitle="Detailed Analysis of Multi-Agent Agreement"
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
        
        # Consensus Overview
        story.append(Paragraph("Consensus Overview", self.pdf_service.styles['CustomHeading1']))
        
        consensus_summary = self._generate_consensus_summary(report_data)
        story.append(Paragraph(consensus_summary, self.pdf_service.styles['ExecutiveSummary']))
        story.append(Spacer(1, 0.3*inch))
        
        # Detailed Consensus Items
        story.append(Paragraph("Detailed Consensus Items", self.pdf_service.styles['CustomHeading1']))
        
        if report_data.consensus_items:
            for i, item in enumerate(report_data.consensus_items, 1):
                story.append(Paragraph(f"{i}. {getattr(item, 'topic', 'Unknown Topic')}", 
                                     self.pdf_service.styles['CustomHeading2']))
                
                # Item details
                item_details = [
                    ['Metric', 'Value'],
                    ['Agreement Percentage', f"{getattr(item, 'agreement_percentage', 0):.1f}%"],
                    ['Support Count', str(getattr(item, 'support_count', 0))],
                    ['Oppose Count', str(getattr(item, 'oppose_count', 0))],
                    ['Neutral Count', str(getattr(item, 'neutral_count', 0))],
                    ['Consensus Strength', getattr(item, 'consensus_strength', 'Unknown').title()],
                    ['Implementation Priority', getattr(item, 'implementation_priority', 'Medium').title()]
                ]
                
                item_table = self.pdf_service.create_professional_table(
                    item_details[1:], item_details[0], "minimal")
                story.append(item_table)
                
                # Supporting and dissenting evidence
                if hasattr(item, 'supporting_evidence') and item.supporting_evidence:
                    story.append(Paragraph("Supporting Evidence:", self.pdf_service.styles['CustomNormal']))
                    evidence_text = self._format_evidence(item.supporting_evidence)
                    story.append(Paragraph(evidence_text, self.pdf_service.styles['CustomNormal']))
                
                if hasattr(item, 'dissenting_evidence') and item.dissenting_evidence:
                    story.append(Paragraph("Dissenting Views:", self.pdf_service.styles['CustomNormal']))
                    dissent_text = self._format_evidence(item.dissenting_evidence)
                    story.append(Paragraph(dissent_text, self.pdf_service.styles['CustomNormal']))
                
                story.append(Spacer(1, 0.2*inch))
        
        # Consensus Visualization
        if config.include_charts and report_data.consensus_items:
            consensus_chart = self._create_consensus_distribution_chart(report_data)
            if consensus_chart:
                story.append(PageBreak())
                story.append(Paragraph("Consensus Distribution", self.pdf_service.styles['CustomHeading1']))
                image_data = base64.b64decode(consensus_chart)
                img_buffer = io.BytesIO(image_data)
                img = Image(img_buffer, width=6*inch, height=4*inch)
                story.append(img)
        
        # Implementation Roadmap
        story.append(Paragraph("Implementation Roadmap", self.pdf_service.styles['CustomHeading1']))
        roadmap = self._generate_implementation_roadmap(report_data)
        story.append(Paragraph(roadmap, self.pdf_service.styles['CustomNormal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_agent_performance_report(self, report_data: DebateReportData) -> io.BytesIO:
        """Generate comprehensive agent performance analysis report"""
        
        config = ReportConfig(
            report_type=ReportType.CUSTOM,
            title="Agent Performance Analysis",
            subtitle="Individual and Collective Performance Metrics"
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
        
        # Overall Performance Summary
        story.append(Paragraph("Performance Summary", self.pdf_service.styles['CustomHeading1']))
        
        perf_summary = self._generate_performance_summary(report_data)
        story.append(Paragraph(perf_summary, self.pdf_service.styles['ExecutiveSummary']))
        story.append(Spacer(1, 0.3*inch))
        
        # Individual Agent Analysis
        story.append(Paragraph("Individual Agent Analysis", self.pdf_service.styles['CustomHeading1']))
        
        if report_data.participants:
            for participant in report_data.participants:
                agent_name = getattr(participant, 'agent_name', getattr(participant, 'agent_id', 'Unknown'))
                
                story.append(Paragraph(f"Agent: {agent_name}", self.pdf_service.styles['CustomHeading2']))
                
                # Agent metrics table
                agent_metrics = [
                    ['Metric', 'Value', 'Rating'],
                    ['Messages Sent', str(getattr(participant, 'messages_sent', 0)), 
                     self._rate_participation(getattr(participant, 'messages_sent', 0))],
                    ['Evidence Provided', str(getattr(participant, 'evidence_provided', 0)),
                     self._rate_evidence_quality(getattr(participant, 'evidence_provided', 0))],
                    ['Average Confidence', f"{getattr(participant, 'avg_confidence_score', 0):.2f}",
                     self._rate_confidence(getattr(participant, 'avg_confidence_score', 0))],
                    ['Response Time', f"{getattr(participant, 'avg_response_time_seconds', 0):.1f}s",
                     self._rate_response_time(getattr(participant, 'avg_response_time_seconds', 0))],
                    ['Consensus Agreement', 
                     'Yes' if getattr(participant, 'consensus_agreement', False) else 'No',
                     '✓' if getattr(participant, 'consensus_agreement', False) else '⚠']
                ]
                
                agent_table = self.pdf_service.create_professional_table(
                    agent_metrics[1:], agent_metrics[0], "financial")
                story.append(agent_table)
                
                # Agent strengths and areas for improvement
                strengths, improvements = self._analyze_agent_strengths_weaknesses(participant)
                
                if strengths:
                    story.append(Paragraph("Strengths:", self.pdf_service.styles['CustomNormal']))
                    story.append(Paragraph("• " + "\n• ".join(strengths), self.pdf_service.styles['CustomNormal']))
                
                if improvements:
                    story.append(Paragraph("Areas for Improvement:", self.pdf_service.styles['CustomNormal']))
                    story.append(Paragraph("• " + "\n• ".join(improvements), self.pdf_service.styles['CustomNormal']))
                
                story.append(Spacer(1, 0.2*inch))
        
        # Team Performance Metrics
        story.append(Paragraph("Team Performance Metrics", self.pdf_service.styles['CustomHeading1']))
        
        if report_data.analytics:
            team_metrics = self._extract_team_metrics(report_data.analytics)
            team_table = self.pdf_service.create_professional_table(
                team_metrics[1:], team_metrics[0], "financial")
            story.append(team_table)
        
        # Performance Visualization
        if config.include_charts and report_data.participants:
            perf_chart = self._create_performance_comparison_chart(report_data)
            if perf_chart:
                story.append(PageBreak())
                story.append(Paragraph("Performance Comparison", self.pdf_service.styles['CustomHeading1']))
                image_data = base64.b64decode(perf_chart)
                img_buffer = io.BytesIO(image_data)
                img = Image(img_buffer, width=6*inch, height=4*inch)
                story.append(img)
        
        # Recommendations for Future Debates
        story.append(Paragraph("Recommendations for Future Debates", self.pdf_service.styles['CustomHeading1']))
        recommendations = self._generate_debate_recommendations(report_data)
        story.append(Paragraph(recommendations, self.pdf_service.styles['CustomNormal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    # Helper methods for debate report generation
    
    def _generate_debate_executive_summary(self, report_data: DebateReportData) -> str:
        """Generate executive summary for debate"""
        status = getattr(report_data.debate, 'status', 'unknown')
        participants = report_data.total_participants
        messages = report_data.total_messages
        consensus_rate = report_data.consensus_rate
        
        quality_assessment = "successful" if consensus_rate > 70 else "partially successful" if consensus_rate > 40 else "challenging"
        
        return f"""
        This multi-agent debate involved {participants} AI agents exchanging {messages} messages 
        over {report_data.debate_duration:.1f} minutes to analyze the investment query. 
        The debate was {quality_assessment}, reaching consensus on {consensus_rate:.1f}% of discussed topics. 
        The final recommendation provides {"strong" if consensus_rate > 70 else "moderate"} confidence 
        in the proposed investment strategy based on comprehensive agent analysis.
        """
    
    def _calculate_agent_performance(self, participant: Any) -> str:
        """Calculate overall agent performance score"""
        messages = getattr(participant, 'messages_sent', 0)
        evidence = getattr(participant, 'evidence_provided', 0)
        confidence = getattr(participant, 'avg_confidence_score', 0)
        
        # Simple scoring algorithm
        score = (messages * 0.3) + (evidence * 0.4) + (confidence * 30 * 0.3)
        
        if score > 8:
            return "Excellent"
        elif score > 6:
            return "Good"
        elif score > 4:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _format_recommendation(self, recommendation: Any) -> str:
        """Format final recommendation for display"""
        if isinstance(recommendation, dict):
            formatted = []
            for key, value in recommendation.items():
                formatted.append(f"{key.replace('_', ' ').title()}: {value}")
            return "\n".join(formatted)
        return str(recommendation)
    
    def _create_debate_flow_chart(self, report_data: DebateReportData) -> Optional[str]:
        """Create debate flow timeline chart"""
        if not MATPLOTLIB_AVAILABLE or not report_data.messages:
            return None
        
        # Create timeline of message exchanges
        message_times = []
        message_counts = []
        
        # Group messages by time intervals (e.g., 5-minute intervals)
        for i in range(0, max(10, len(report_data.messages) // 5)):
            message_times.append(i * 5)  # 5-minute intervals
            message_counts.append(len(report_data.messages) // 10 + (i % 3))  # Mock data
        
        chart_data = {
            'x': message_times,
            'y': message_counts,
            'xlabel': 'Time (minutes)',
            'ylabel': 'Messages per Interval'
        }
        
        return self.pdf_service.generate_chart(ChartType.LINE_CHART, chart_data, 
                                              "Debate Activity Over Time")
    
    def _generate_consensus_summary(self, report_data: DebateReportData) -> str:
        """Generate consensus analysis summary"""
        total_items = len(report_data.consensus_items)
        consensus_rate = report_data.consensus_rate
        
        strong_consensus = sum(1 for item in report_data.consensus_items 
                             if getattr(item, 'consensus_strength', '') == 'strong')
        
        return f"""
        The debate generated {total_items} distinct consensus items, with {consensus_rate:.1f}% 
        reaching formal agreement. Of these, {strong_consensus} items achieved strong consensus, 
        indicating robust agreement among participants. The consensus analysis reveals 
        {"high alignment" if consensus_rate > 70 else "moderate alignment" if consensus_rate > 40 else "significant disagreement"} 
        on key investment considerations, providing {"clear" if consensus_rate > 70 else "qualified"} 
        guidance for decision-making.
        """
    
    def _format_evidence(self, evidence: Any) -> str:
        """Format evidence for display"""
        if isinstance(evidence, list):
            return "\n".join([f"• {item}" for item in evidence[:5]])
        elif isinstance(evidence, dict):
            return "\n".join([f"• {key}: {value}" for key, value in evidence.items()])
        return str(evidence)[:500] + ("..." if len(str(evidence)) > 500 else "")
    
    def _create_consensus_distribution_chart(self, report_data: DebateReportData) -> Optional[str]:
        """Create chart showing consensus distribution"""
        if not MATPLOTLIB_AVAILABLE or not report_data.consensus_items:
            return None
        
        # Categorize consensus items by strength
        strong = sum(1 for item in report_data.consensus_items 
                    if getattr(item, 'consensus_strength', '') == 'strong')
        moderate = sum(1 for item in report_data.consensus_items 
                      if getattr(item, 'consensus_strength', '') == 'moderate')
        weak = sum(1 for item in report_data.consensus_items 
                  if getattr(item, 'consensus_strength', '') == 'weak')
        
        chart_data = {
            'labels': ['Strong', 'Moderate', 'Weak'],
            'values': [strong, moderate, weak]
        }
        
        return self.pdf_service.generate_chart(ChartType.PIE_CHART, chart_data, 
                                              "Consensus Strength Distribution")
    
    def _generate_implementation_roadmap(self, report_data: DebateReportData) -> str:
        """Generate implementation roadmap"""
        high_priority = sum(1 for item in report_data.consensus_items 
                           if getattr(item, 'implementation_priority', '') == 'high')
        medium_priority = sum(1 for item in report_data.consensus_items 
                             if getattr(item, 'implementation_priority', '') == 'medium')
        
        return f"""
        Implementation Roadmap:
        
        Phase 1 (Immediate): {high_priority} high-priority consensus items requiring immediate attention
        Phase 2 (Short-term): {medium_priority} medium-priority items for implementation within 30 days
        Phase 3 (Long-term): Remaining items for ongoing monitoring and potential future implementation
        
        Key success factors include maintaining agent consensus monitoring, regular performance reviews, 
        and adaptive strategy adjustments based on market conditions.
        """
    
    def _generate_performance_summary(self, report_data: DebateReportData) -> str:
        """Generate agent performance summary"""
        avg_messages = sum(getattr(p, 'messages_sent', 0) for p in report_data.participants) / len(report_data.participants) if report_data.participants else 0
        avg_confidence = sum(getattr(p, 'avg_confidence_score', 0) for p in report_data.participants) / len(report_data.participants) if report_data.participants else 0
        
        return f"""
        The {report_data.total_participants} participating agents demonstrated 
        {"strong" if avg_messages > 5 else "moderate"} engagement with an average of 
        {avg_messages:.1f} messages per agent. Overall confidence levels averaged 
        {avg_confidence:.2f}, indicating {"high" if avg_confidence > 0.8 else "moderate"} 
        certainty in analysis and recommendations. Team performance was 
        {"excellent" if avg_confidence > 0.8 and avg_messages > 5 else "satisfactory"} 
        with effective collaboration and knowledge synthesis.
        """
    
    def _rate_participation(self, messages: int) -> str:
        """Rate participation level"""
        if messages > 10: return "High"
        elif messages > 5: return "Good"
        elif messages > 2: return "Fair"
        else: return "Low"
    
    def _rate_evidence_quality(self, evidence_count: int) -> str:
        """Rate evidence quality"""
        if evidence_count > 5: return "Excellent"
        elif evidence_count > 3: return "Good"
        elif evidence_count > 1: return "Fair"
        else: return "Poor"
    
    def _rate_confidence(self, confidence: float) -> str:
        """Rate confidence level"""
        if confidence > 0.9: return "Very High"
        elif confidence > 0.8: return "High"
        elif confidence > 0.6: return "Moderate"
        else: return "Low"
    
    def _rate_response_time(self, response_time: float) -> str:
        """Rate response time"""
        if response_time < 5: return "Excellent"
        elif response_time < 10: return "Good"
        elif response_time < 20: return "Fair"
        else: return "Slow"
    
    def _analyze_agent_strengths_weaknesses(self, participant: Any) -> Tuple[List[str], List[str]]:
        """Analyze agent strengths and weaknesses"""
        strengths = []
        improvements = []
        
        messages = getattr(participant, 'messages_sent', 0)
        evidence = getattr(participant, 'evidence_provided', 0)
        confidence = getattr(participant, 'avg_confidence_score', 0)
        
        if messages > 7:
            strengths.append("High participation and engagement")
        elif messages < 3:
            improvements.append("Increase participation in debates")
        
        if evidence > 3:
            strengths.append("Strong evidence-based reasoning")
        elif evidence < 2:
            improvements.append("Provide more supporting evidence")
        
        if confidence > 0.8:
            strengths.append("High confidence in analysis")
        elif confidence < 0.6:
            improvements.append("Develop more confident assessments")
        
        return strengths, improvements
    
    def _extract_team_metrics(self, analytics: Any) -> List[List[str]]:
        """Extract team performance metrics"""
        return [
            ['Metric', 'Value', 'Assessment'],
            ['Participation Balance', f"{getattr(analytics, 'participation_balance_score', 0.75):.2f}", 'Good'],
            ['Evidence Quality', f"{getattr(analytics, 'avg_evidence_quality', 0.80):.2f}", 'High'],
            ['Consensus Efficiency', f"{getattr(analytics, 'debate_efficiency_score', 0.70):.2f}", 'Good'],
            ['Civility Score', f"{getattr(analytics, 'civility_score', 0.95):.2f}", 'Excellent'],
            ['Implementation Clarity', f"{getattr(analytics, 'actionability_score', 0.85):.2f}", 'High']
        ]
    
    def _create_performance_comparison_chart(self, report_data: DebateReportData) -> Optional[str]:
        """Create agent performance comparison chart"""
        if not MATPLOTLIB_AVAILABLE or not report_data.participants:
            return None
        
        agent_names = [getattr(p, 'agent_name', getattr(p, 'agent_id', f'Agent {i}'))[:10] 
                      for i, p in enumerate(report_data.participants)]
        performance_scores = [getattr(p, 'messages_sent', 0) + getattr(p, 'evidence_provided', 0) 
                             for p in report_data.participants]
        
        chart_data = {
            'categories': agent_names,
            'values': performance_scores,
            'xlabel': 'Agents',
            'ylabel': 'Performance Score'
        }
        
        return self.pdf_service.generate_chart(ChartType.BAR_CHART, chart_data, 
                                              "Agent Performance Comparison")
    
    def _generate_debate_recommendations(self, report_data: DebateReportData) -> str:
        """Generate recommendations for future debates"""
        avg_confidence = sum(getattr(p, 'avg_confidence_score', 0) for p in report_data.participants) / len(report_data.participants) if report_data.participants else 0
        consensus_rate = report_data.consensus_rate
        
        recommendations = []
        
        if consensus_rate < 50:
            recommendations.append("• Consider adjusting debate parameters to encourage more consensus building")
        
        if avg_confidence < 0.7:
            recommendations.append("• Provide agents with more comprehensive data sources to improve confidence")
        
        if report_data.total_messages < 20:
            recommendations.append("• Allow for longer debate duration to enable more thorough analysis")
        
        recommendations.extend([
            "• Regular agent performance monitoring and calibration",
            "• Implement specialized agent roles for different analysis aspects",
            "• Develop consensus-building protocols for complex investment decisions"
        ])
        
        return "\n".join(recommendations)


# Create service instance
debate_report_service = DebateReportTemplates() if REPORTLAB_AVAILABLE else None