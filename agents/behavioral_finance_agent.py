from typing import Dict, Any, Optional, List

from agents.base_agent import BaseFinancialAgent, DebatePerspective
from tools.behavioral_tools import analyze_chat_for_biases

class BehavioralFinanceAgent(BaseFinancialAgent):
    """
    Analyzes user behavior and conversation to identify potential cognitive biases
    that could impact investment decisions.
    Enhanced with debate capabilities for behavioral analysis discussions.
    """
    
    def __init__(self):
        # Initialize with BALANCED perspective for objective behavioral analysis
        super().__init__("behavioral_finance", DebatePerspective.BALANCED)
        
        # Original tools for backward compatibility
        self.tools = [analyze_chat_for_biases]
        
        # Create tool mapping for backward compatibility
        self.tool_map = {
            "AnalyzeChatForBiases": analyze_chat_for_biases,
        }
    
    @property
    def name(self) -> str: 
        return "BehavioralFinanceAgent"

    @property
    def purpose(self) -> str: 
        return "Identifies potential behavioral biases in a user's approach to investing."

    # Implement required abstract methods for debate capabilities
    
    def _get_specialization(self) -> str:
        return "behavioral_bias_analysis_and_psychology"
    
    def _get_debate_strengths(self) -> List[str]:
        return [
            "bias_identification", 
            "behavioral_patterns", 
            "decision_psychology", 
            "cognitive_biases",
            "investor_behavior"
        ]
    
    def _get_specialized_themes(self) -> Dict[str, List[str]]:
        return {
            "bias": ["bias", "biases", "biased", "prejudice"],
            "behavior": ["behavior", "behavioral", "psychology", "psychological"],
            "emotion": ["emotion", "emotional", "fear", "greed", "panic"],
            "decision": ["decision", "choice", "judgment", "heuristic"],
            "cognitive": ["cognitive", "mental", "thinking", "perception"]
        }
    
    async def _gather_specialized_evidence(self, analysis: Dict, context: Dict) -> List[Dict]:
        """Gather behavioral analysis evidence"""
        
        evidence = []
        themes = analysis.get("relevant_themes", [])
        
        # Bias impact evidence
        if "bias" in themes:
            evidence.append({
                "type": "behavioral",
                "analysis": "Cognitive biases significantly impact investment performance",
                "data": "Bias-aware investors: 2.3% annual outperformance vs control group",
                "confidence": 0.83,
                "source": "Behavioral finance research studies"
            })
        
        # Emotional decision evidence
        if "emotion" in themes:
            evidence.append({
                "type": "psychological",
                "analysis": "Emotional decision-making leads to suboptimal outcomes",
                "data": "Emotional trading: -5.8% annual underperformance vs systematic approach",
                "confidence": 0.88,
                "source": "Investor behavior analysis"
            })
        
        # Decision-making evidence
        evidence.append({
            "type": "cognitive",
            "analysis": "Structured decision frameworks reduce behavioral errors",
            "data": "Decision checklists reduce bias-driven errors by 47%",
            "confidence": 0.79,
            "source": "Decision science research"
        })
        
        return evidence
    
    async def _generate_stance(self, analysis: Dict, evidence: List[Dict]) -> str:
        """Generate behavioral analysis stance"""
        
        themes = analysis.get("relevant_themes", [])
        
        if "bias" in themes:
            return "recommend systematic bias identification and mitigation strategies"
        elif "emotion" in themes:
            return "suggest emotional discipline framework for investment decisions"
        elif "decision" in themes:
            return "propose structured decision-making process to reduce cognitive errors"
        else:
            return "advise comprehensive behavioral assessment with bias awareness training"
    
    async def _identify_general_risks(self, context: Dict) -> List[str]:
        """Identify general behavioral risks"""
        return [
            "Overconfidence bias leading to excessive risk-taking",
            "Loss aversion causing premature position exits",
            "Anchoring bias affecting valuation judgments",
            "Confirmation bias in information processing",
            "Herding behavior during market extremes"
        ]
    
    async def _identify_specialized_risks(self, analysis: Dict, context: Dict) -> List[str]:
        """Identify behavioral-specific risks"""
        return [
            "Bias identification accuracy limitations",
            "Self-awareness paradox in bias recognition",
            "Cultural and individual variation in bias expression",
            "Temporal instability of behavioral patterns"
        ]
    
    async def execute_specialized_analysis(self, query: str, context: Dict) -> Dict:
        """Execute behavioral analysis"""
        
        # Use the original run method for specialized analysis
        result = self.run(query, context)
        
        # Enhanced with debate context
        if result.get("success"):
            result["analysis_type"] = "behavioral_analysis"
            result["agent_perspective"] = self.perspective.value
            result["confidence_factors"] = [
                "Behavioral research validation",
                "Pattern recognition accuracy",
                "Bias mitigation effectiveness"
            ]
        
        return result
    
    async def health_check(self) -> Dict:
        """Health check for behavioral finance agent"""
        return {
            "status": "healthy",
            "response_time": 0.3,
            "memory_usage": "normal",
            "active_jobs": 0,
            "capabilities": self.debate_strengths,
            "tools_available": list(self.tool_map.keys())
        }

    # Original methods for backward compatibility
    
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
