# in agents/orchestrator.py
import re
from typing import Dict, Any, Set, Optional

# Import all the specialist agents
from agents.quantitative_analyst import QuantitativeAnalystAgent
from agents.strategy_architect import StrategyArchitectAgent
from agents.strategy_rebalancing import StrategyRebalancingAgent
from agents.hedging_strategist import HedgingStrategistAgent
from agents.strategy_backtester import StrategyBacktesterAgent

# Import our new database and data handling utilities
from db import crud
from core.data_handler import get_market_data_for_portfolio

class FinancialOrchestrator:
    """
    ### UPGRADED: Now connects to the database to provide real user context to agents.
    """
    def __init__(self):
        self.roster = {
            "QuantitativeAnalyst": QuantitativeAnalystAgent(),
            "StrategyArchitect": StrategyArchitectAgent(),
            "StrategyRebalancing": StrategyRebalancingAgent(),
            "HedgingStrategist": HedgingStrategistAgent(),
            "StrategyBacktester": StrategyBacktesterAgent(),
        }
        print("FinancialOrchestrator initialized with a team of 5 agents.")
        self._setup_classification_patterns()

    def _setup_classification_patterns(self):
        """Setup pattern-based query classification using keyword sets for robustness."""
        self.routing_map = {
            "QuantitativeAnalyst": {frozenset(["risk"]), frozenset(["report"]), frozenset(["factor"]), frozenset(["alpha"]), frozenset(["cvar"])},
            "StrategyArchitect": {frozenset(["design"]), frozenset(["find", "strategy"]), frozenset(["mean-reversion"])},
            "StrategyRebalancing": {frozenset(["rebalance"]), frozenset(["optimize"]), frozenset(["allocation"]), frozenset(["risk", "parity"]), frozenset(["diversify", "risk"])},
            "HedgingStrategist": {frozenset(["hedge"]), frozenset(["protect"]), frozenset(["target", "volatility"])},
            "StrategyBacktester": {frozenset(["backtest"]), frozenset(["test", "strategy"])}
        }

    def _classify_query(self, query_words: Set[str]) -> Optional[str]:
        """Classifies the query and returns the name of the best agent to handle it."""
        agent_scores = {name: 0 for name in self.roster.keys()}
        for agent_name, keyword_sets in self.routing_map.items():
            for keyword_set in keyword_sets:
                if keyword_set.issubset(query_words):
                    agent_scores[agent_name] += len(keyword_set)
        if max(agent_scores.values()) > 0:
            return max(agent_scores, key=agent_scores.get)
        return None

    def route_query(self, user_query: str, db_session, current_user) -> Dict[str, Any]:
        """
        The main entry point. It now fetches real user data before routing.
        """
        # 1. Classify the query
        clean_query = re.sub(r'[^\w\s]', '', user_query.lower())
        query_words = set(clean_query.split())
        selected_agent_name = self._classify_query(query_words)

        if not selected_agent_name:
            print("Orchestrator could not determine a route.")
            return {"success": False, "error": "I'm not sure which specialist can handle that request."}

        # 2. Fetch the user's primary portfolio from the database
        user_portfolios = crud.get_user_portfolios(db=db_session, user_id=current_user.id)
        if not user_portfolios:
            # Handle agents that don't need a portfolio (like the architect)
            if selected_agent_name in ["StrategyArchitect"]:
                 agent_to_run = self.roster[selected_agent_name]
                 print(f"Orchestrator routing query to: {agent_to_run.name} (no portfolio context needed)")
                 return agent_to_run.run(user_query, context={}) # Pass empty context
            return {"success": False, "error": "You don't have a portfolio yet. Please create one first."}

        primary_portfolio = user_portfolios[0] # For now, we just use the user's first portfolio

        # 3. Use the data handler to get full market context
        print(f"Fetching market data for portfolio ID: {primary_portfolio.id}")
        portfolio_context = get_market_data_for_portfolio(primary_portfolio.holdings)
        
        # Add user and portfolio objects to context for agents that might need them
        portfolio_context['user'] = current_user
        portfolio_context['portfolio_model'] = primary_portfolio
        
        # 4. Route to the selected agent with the rich context
        agent_to_run = self.roster[selected_agent_name]
        print(f"Orchestrator routing query to: {agent_to_run.name}")
        return agent_to_run.run(user_query, context=portfolio_context)