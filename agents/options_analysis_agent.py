# agents/options_analysis_agent.py
"""
OptionsAnalysisAgent - Advanced Options Strategy & Analysis
============================================================
This agent specializes in quantitative analysis of options, including
pricing, risk assessment (Greeks), and strategy generation. It is designed
to participate in debates with a focus on risk management and volatility.

Enhanced with more sophisticated strategy recommendations and portfolio
risk analysis capabilities.
"""

from agents.mcp_base_agent import MCPBaseAgent
from typing import Dict, List, Any
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

class OptionsAnalysisAgent(MCPBaseAgent):
    """An AI agent for sophisticated options analysis and strategy."""

    def __init__(self, agent_id: str = "options_analyst"):
        super().__init__(
            agent_id=agent_id,
            agent_name="Options Analysis Agent",
            capabilities=[
                "options_pricing",
                "calculate_greeks",
                "volatility_analysis",
                "strategy_recommendation",
                "portfolio_risk_analysis", # New capability
                "debate_participation"
            ]
        )
        self.specialization = "options_strategy_and_risk_management"

    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict:
        """Execute a specific options analysis capability."""
        logger.info(f"Executing options capability: {capability}")
        
        capability_map = {
            "options_pricing": self.price_option,
            "calculate_greeks": self.calculate_greeks,
            "volatility_analysis": self.analyze_volatility,
            "strategy_recommendation": self.recommend_strategy,
            "portfolio_risk_analysis": self.analyze_portfolio_risk,
        }
        
        if capability not in capability_map:
            return {"error": f"Capability '{capability}' not supported."}
            
        return await capability_map[capability](data, context)

    # --- Core Capabilities ---

    async def price_option(self, data: Dict, context: Dict) -> Dict:
        """Calculates the theoretical price of an option using the Black-Scholes model."""
        try:
            # Placeholder for Volatility Smile/Skew adjustment
            # volatility = self._get_skew_adjusted_volatility(data['strike_price'], ...)
            price = self._black_scholes(**data)
            return {"theoretical_price": price, "model": "Black-Scholes"}
        except (TypeError, KeyError) as e:
            return {"error": f"Missing required data for pricing: {e}"}

    async def calculate_greeks(self, data: Dict, context: Dict) -> Dict:
        """Calculates the primary option Greeks for risk management."""
        try:
            greeks = self._calculate_all_greeks(**data)
            return greeks
        except (TypeError, KeyError) as e:
            return {"error": f"Missing required data for Greeks calculation: {e}"}

    async def analyze_volatility(self, data: Dict, context: Dict) -> Dict:
        """Analyzes the implied vs. historical volatility for a given underlying."""
        symbol = data.get("symbol")
        hv = data.get("historical_volatility")
        iv = data.get("implied_volatility")

        if not all([symbol, hv, iv]):
            return {"error": "Missing symbol, historical_volatility, or implied_volatility."}

        volatility_premium = iv - hv
        assessment = "rich" if volatility_premium > 0.02 else "cheap" if volatility_premium < -0.02 else "fair"
        
        return {
            "symbol": symbol,
            "historical_volatility": hv,
            "implied_volatility": iv,
            "volatility_premium_ratio": volatility_premium / hv if hv > 0 else 0,
            "assessment": f"Implied volatility is currently {assessment} compared to historical levels.",
            "strategy_implication": "Consider selling premium (e.g., covered calls, cash-secured puts)." if assessment == "rich" else "Consider buying premium (e.g., long calls/puts)."
        }

    async def recommend_strategy(self, data: Dict, context: Dict) -> Dict:
        """
        Recommends an options strategy based on market outlook, existing positions, and risk tolerance.
        """
        outlook = data.get("market_outlook")
        risk = data.get("risk_tolerance")
        existing_position = data.get("existing_position") # e.g., {"type": "stock", "shares": 100}

        if not outlook or not risk:
            return {"error": "Missing market_outlook or risk_tolerance."}
        
        # Enhanced strategy selection logic
        if existing_position and existing_position.get("type") == "stock" and existing_position.get("shares", 0) >= 100:
            if outlook == 'neutral' or outlook == 'moderately_bullish':
                return {"name": "Covered Call", "description": "Sell a call option against your existing stock holding to generate income. Best if you expect the stock to remain flat or rise slightly."}
        
        if outlook == 'bearish' and risk == 'conservative' and existing_position:
             return {"name": "Protective Put", "description": "Buy a put option to hedge against a potential decline in your existing stock holding. Acts like insurance."}

        strategy_map = {
            ('bullish', 'aggressive'): {"name": "Long Call", "description": "Buy a call option to speculate on a sharp price increase."},
            ('bullish', 'moderate'): {"name": "Bull Call Spread", "description": "Buy a call and sell a higher-strike call to limit cost and risk."},
            ('bearish', 'aggressive'): {"name": "Long Put", "description": "Buy a put option to speculate on a sharp price decrease."},
            ('bearish', 'moderate'): {"name": "Bear Put Spread", "description": "Buy a put and sell a lower-strike put to limit cost and risk."},
            ('neutral', 'moderate'): {"name": "Iron Condor", "description": "A defined-risk strategy that profits if the stock stays within a range."},
            ('volatile', 'aggressive'): {"name": "Long Straddle", "description": "Buy a call and a put at the same strike to profit from a large price move in either direction."},
        }
        
        recommendation = strategy_map.get((outlook, risk), {"name": "Hold/No Action", "description": "Current conditions do not strongly favor a new options position based on the inputs."})
        return recommendation

    async def analyze_portfolio_risk(self, data: Dict, context: Dict) -> Dict:
        """
        Analyzes the risk of a proposed options position in the context of the overall portfolio.
        """
        try:
            position_value = data["position_value"]
            portfolio_value = data["portfolio_value"]
            max_risk_percent = data.get("max_risk_percent", 2.0) # Default to 2% max risk

            return self._calculate_position_risk(position_value, portfolio_value, max_risk_percent)
        except KeyError as e:
            return {"error": f"Missing required data for portfolio risk analysis: {e}"}

    # --- Mathematical & Risk Helper Methods ---

    def _calculate_position_risk(self, position_value: float, portfolio_value: float, max_risk_percent: float = 2.0) -> Dict:
        """Calculates if a position size is appropriate based on portfolio risk tolerance."""
        if portfolio_value <= 0:
            return {"error": "Portfolio value must be positive."}
            
        position_risk_pct = (position_value / portfolio_value) * 100
        max_allowed_value = (max_risk_percent / 100) * portfolio_value
        
        is_within_limit = position_risk_pct <= max_risk_percent
        
        return {
            "position_value": position_value,
            "portfolio_value": portfolio_value,
            "position_risk_of_portfolio": round(position_risk_pct, 2),
            "max_risk_tolerance_pct": max_risk_percent,
            "is_within_risk_limit": is_within_limit,
            "max_recommended_position_value": round(max_allowed_value, 2),
            "assessment": "Position size is within the defined risk tolerance." if is_within_limit else "Warning: Position size exceeds the defined risk tolerance."
        }
        
    def _black_scholes(self, stock_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type):
        """Core Black-Scholes-Merton calculation."""
        S, K, T, r, sigma = stock_price, strike_price, time_to_expiry, risk_free_rate, volatility
        if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        elif option_type == 'put':
            price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")
        return price

    def _calculate_all_greeks(self, stock_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type):
        """Calculates Delta, Gamma, Vega, Theta, and Rho."""
        S, K, T, r, sigma = stock_price, strike_price, time_to_expiry, risk_free_rate, volatility
        if T <= 0: return {"delta": 1 if S > K else 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        
        greeks = {
            "gamma": pdf_d1 / (S * sigma * np.sqrt(T)),
            "vega": S * pdf_d1 * np.sqrt(T) / 100,
        }
        
        if option_type == 'call':
            greeks["delta"] = norm.cdf(d1)
            greeks["theta"] = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            greeks["rho"] = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        elif option_type == 'put':
            greeks["delta"] = -norm.cdf(-d1)
            greeks["theta"] = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            greeks["rho"] = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")
            
        return greeks
