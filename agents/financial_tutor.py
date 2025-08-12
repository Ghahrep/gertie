# in agents/financial_tutor.py
from typing import Dict, Any, Optional
from agents.base_agent import BaseFinancialAgent
import re

class FinancialTutorAgent(BaseFinancialAgent):
    """
    Acts as a knowledge translator, explaining complex financial concepts to the user.
    """
    
    @property
    def name(self) -> str: return "FinancialTutor"

    @property
    def purpose(self) -> str: return "Explains financial concepts, terms, and strategies in an easy-to-understand way."

    def __init__(self):
        # This agent starts with no complex tools, its knowledge is internal.
        tools = []
        super().__init__(tools)
        
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