# in agents/behavioral_finance_agent.py
from typing import Dict, Any, Optional, List
from agents.base_agent import BaseFinancialAgent
from tools.behavioral_tools import analyze_chat_for_biases

class BehavioralFinanceAgent(BaseFinancialAgent):
    """
    Analyzes user behavior and conversation to identify potential cognitive biases
    that could impact investment decisions.
    """
    @property
    def name(self) -> str: return "BehavioralFinanceAgent"

    @property
    def purpose(self) -> str: return "Identifies potential behavioral biases in a user's approach to investing."

    def __init__(self):
        tools = [analyze_chat_for_biases]
        super().__init__(tools)

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        
        # This agent needs conversation history, which the orchestrator will need to provide.
        # For now, we'll simulate it. In a real implementation, this would come from the context.
        chat_history = context.get("chat_history", [])
        if not chat_history:
            return {
                "success": True,
                "summary": "I can analyze your conversation for potential behavioral biases, but I need more conversation history to provide an analysis.",
                "agent_used": self.name
            }

        print("Executing tool 'AnalyzeChatForBiases'...")
        analysis_tool = self.tool_map["AnalyzeChatForBiases"]
        result = analysis_tool.invoke({"chat_history": chat_history})
        
        # Format the summary
        if result.get("biases_detected"):
            summary = "Based on our conversation, I've noticed a few potential behavioral patterns:\n"
            for bias, details in result["biases_detected"].items():
                summary += f"\n- **{bias}:** {details['finding']}\n  - *Suggestion:* {details['suggestion']}\n"
        else:
            summary = result.get("summary", "No significant behavioral patterns were detected.")

        result['summary'] = summary
        result['agent_used'] = self.name
        return result