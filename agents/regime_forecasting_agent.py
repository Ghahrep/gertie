# in agents/regime_forecasting_agent.py
from typing import Dict, Any, Optional
from agents.base_agent import BaseFinancialAgent

# Import the tools this agent will use
from tools.regime_tools import detect_hmm_regimes, forecast_regime_transition_probability

class RegimeForecastingAgent(BaseFinancialAgent):
    """
    Analyzes historical data to detect the current market regime and forecasts the
    probability of transitioning to other regimes.
    """
    @property
    def name(self) -> str: return "RegimeForecastingAgent"

    @property
    def purpose(self) -> str: return "Detects current market regimes and forecasts transition probabilities."

    def __init__(self):
        tools = [
            detect_hmm_regimes,
            forecast_regime_transition_probability
        ]
        super().__init__(tools)

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"--- {self.name} Agent Received Query: '{user_query}' ---")

        if not context or "portfolio_returns" not in context:
            return {"success": False, "error": "Could not perform regime analysis. Portfolio data is missing."}

        # --- Two-Step Tool Chain ---
        # 1. Detect the current regimes
        print("Executing tool 'DetectHMMRegimes'...")
        detection_tool = self.tool_map["DetectHMMRegimes"]
        detection_result = detection_tool.run({"returns": context["portfolio_returns"]})

        if not detection_result or "fitted_model" not in detection_result:
            return {"success": False, "error": "Failed to detect market regimes."}

        # 2. Forecast the transition probabilities
        print("Executing tool 'ForecastRegimeTransitionProbability'...")
        forecast_tool = self.tool_map["ForecastRegimeTransitionProbability"]
        forecast_result = forecast_tool.run({
            "hmm_results": detection_result # Pass the full result from the first tool
        })

        if not forecast_result:
            return {"success": False, "error": "Failed to forecast regime transitions."}
            
        # 3. Create a user-friendly summary
        current_regime = forecast_result['from_regime']['index']
        probs = forecast_result['transition_forecast']
        summary = f"The portfolio is currently in Regime {current_regime} (Low Volatility is 0, High is 1).\n"
        summary += "The probabilities for the next period are:\n"
        for prob_info in probs:
            summary += f"- Transition to Regime {prob_info['to_regime_index']}: {prob_info['probability']:.1%}\n"
        
        return {
            "success": True,
            "summary": summary,
            "agent_used": self.name,
            "data": forecast_result
        }