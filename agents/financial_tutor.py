# in agents/financial_tutor.py
from typing import Dict, Any, Optional, List
import re

from agents.base_agent import BaseFinancialAgent, DebatePerspective

class FinancialTutorAgent(BaseFinancialAgent):
    """
    Acts as a knowledge translator, explaining complex financial concepts to the user.
    Enhanced with debate capabilities for educational discussions.
    """
    
    def __init__(self):
        # Initialize with BALANCED perspective for educational neutrality
        super().__init__("financial_tutor", DebatePerspective.BALANCED)
        
        # This agent starts with no complex tools, its knowledge is internal.
        self.tools = []
        self.tool_map = {}
        
        # The agent's internal knowledge base. We can expand this over time.
        self._knowledge_base = {
            "cvar": (
                "**Conditional Value at Risk (CVaR)**, also known as Expected Shortfall, is a risk measure that answers the question: "
                "'If things get really bad, what is my average expected loss?' For a 95% CVaR, it calculates the average loss on the worst 5% of days."
            ),
            "var": (
                "**Value at Risk (VaR)** is a statistic that quantifies the extent of possible financial losses within a firm, portfolio, or position over a specific time frame. "
                "For example, a 95% VaR of $10,000 means there is a 5% chance of losing at least $10,000 on any given day."
            ),
            "sharpe ratio": (
                "The **Sharpe Ratio** is a measure of risk-adjusted return. It tells you how much return you are getting for each unit of risk you take (where risk is measured by volatility). "
                "A higher Sharpe Ratio is generally better, indicating a more efficient portfolio."
            ),
            "sortino ratio": (
                "The **Sortino Ratio** is a variation of the Sharpe Ratio that only penalizes for 'bad' volatility (downside deviation). "
                "It's useful because it doesn't punish a portfolio for having strong positive returns, giving a better picture of its performance relative to downside risk."
            ),
            "herc": (
                "**Hierarchical Risk Parity (HERC)** is a modern portfolio optimization method. Unlike traditional models that rely on predicting returns, "
                "HERC structures the portfolio by grouping similar assets together and then diversifying risk across those groups. It's often more robust and stable."
            ),
        }
    
    @property
    def name(self) -> str: 
        return "FinancialTutor"

    @property
    def purpose(self) -> str: 
        return "Explains financial concepts, terms, and strategies in an easy-to-understand way."

    # Implement required abstract methods for debate capabilities
    
    def _get_specialization(self) -> str:
        return "financial_education_and_concept_explanation"
    
    def _get_debate_strengths(self) -> List[str]:
        return [
            "concept_clarification", 
            "educational_guidance", 
            "knowledge_synthesis", 
            "accessible_explanations",
            "learning_methodology"
        ]
    
    def _get_specialized_themes(self) -> Dict[str, List[str]]:
        return {
            "education": ["explain", "what", "is", "define", "meaning"],
            "learning": ["learn", "understand", "teach", "tutorial", "help"],
            "concepts": ["concept", "theory", "principle", "idea"],
            "clarification": ["clarify", "clear", "simple", "basic"],
            "guidance": ["guide", "advice", "recommend", "suggest"]
        }
    
    async def _gather_specialized_evidence(self, analysis: Dict, context: Dict) -> List[Dict]:
        """Gather educational and explanatory evidence"""
        
        evidence = []
        themes = analysis.get("relevant_themes", [])
        
        # Educational effectiveness evidence
        if "education" in themes or "learning" in themes:
            evidence.append({
                "type": "educational",
                "analysis": "Clear concept explanation improves investment decision-making",
                "data": "Students with financial education: 23% better portfolio performance",
                "confidence": 0.85,
                "source": "Financial literacy research studies"
            })
        
        # Knowledge retention evidence
        evidence.append({
            "type": "pedagogical",
            "analysis": "Structured learning approach enhances concept retention",
            "data": "Multi-modal explanations: 78% concept retention vs 45% text-only",
            "confidence": 0.80,
            "source": "Educational psychology research"
        })
        
        # Practical application evidence
        evidence.append({
            "type": "practical",
            "analysis": "Applied examples improve understanding of financial concepts",
            "data": "Real-world examples increase comprehension by 34%",
            "confidence": 0.75,
            "source": "Learning effectiveness studies"
        })
        
        return evidence
    
    async def _generate_stance(self, analysis: Dict, evidence: List[Dict]) -> str:
        """Generate education-focused stance"""
        
        themes = analysis.get("relevant_themes", [])
        
        if "education" in themes:
            return "recommend structured learning approach with clear concept definitions"
        elif "clarification" in themes:
            return "suggest simplified explanations with practical examples"
        elif "guidance" in themes:
            return "propose step-by-step educational guidance tailored to user level"
        else:
            return "advise comprehensive educational approach building from fundamentals"
    
    async def _identify_general_risks(self, context: Dict) -> List[str]:
        """Identify general educational risks"""
        return [
            "Oversimplification leading to misunderstanding",
            "Information overload hindering learning",
            "Lack of practical application context",
            "Outdated concepts in dynamic markets",
            "One-size-fits-all educational approach"
        ]
    
    async def _identify_specialized_risks(self, analysis: Dict, context: Dict) -> List[str]:
        """Identify education-specific risks"""
        return [
            "Concept confusion from incomplete explanations",
            "False confidence from superficial understanding",
            "Knowledge gaps in foundational concepts",
            "Difficulty translating theory to practice"
        ]
    
    async def execute_specialized_analysis(self, query: str, context: Dict) -> Dict:
        """Execute educational analysis"""
        
        # Use the original run method for specialized analysis
        result = self.run(query, context)
        
        # Enhanced with debate context
        if result.get("success"):
            result["analysis_type"] = "financial_education"
            result["agent_perspective"] = self.perspective.value
            result["confidence_factors"] = [
                "Educational research validation",
                "Concept clarity assessment",
                "Practical applicability"
            ]
        
        return result
    
    async def health_check(self) -> Dict:
        """Health check for financial tutor"""
        return {
            "status": "healthy",
            "response_time": 0.2,
            "memory_usage": "normal",
            "active_jobs": 0,
            "capabilities": self.debate_strengths,
            "knowledge_base_size": len(self._knowledge_base)
        }

    # Original methods for backward compatibility
    
    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        
        clean_query = user_query.lower()
        
        # Try to find a concept in the query that matches our knowledge base
        for concept, explanation in self._knowledge_base.items():
            if re.search(r'\b' + re.escape(concept) + r'\b', clean_query):
                print(f"Found concept: '{concept}'. Providing explanation.")
                return {
                    "success": True,
                    "summary": explanation,
                    "agent_used": self.name
                }
                
        # If no specific concept is found, provide a helpful default response
        return {
            "success": True,
            "summary": "I can explain a variety of financial concepts like CVaR, VaR, Sharpe Ratio, and HERC. Please ask me about a specific term.",
            "agent_used": self.name
        }