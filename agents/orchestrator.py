# in agents/orchestrator.py
import re
from typing import Dict, Any, Set, Optional
from db import crud

# Import all the specialist agents
from agents.quantitative_analyst import QuantitativeAnalystAgent
from agents.strategy_architect import StrategyArchitectAgent
from agents.strategy_rebalancing import StrategyRebalancingAgent
from agents.hedging_strategist import HedgingStrategistAgent
from agents.strategy_backtester import StrategyBacktesterAgent
from agents.financial_tutor import FinancialTutorAgent
from agents.regime_forecasting_agent import RegimeForecastingAgent
from core.data_handler import get_market_data_for_portfolio

# We will add this agent in a future step, but can define it now.
# from agents.financial_tutor import FinancialTutorAgent 

class FinancialOrchestrator:
    """
    ### UPGRADED: Uses a more sophisticated, prioritized routing logic.
    """
    def __init__(self):
        self.roster = {
            "QuantitativeAnalyst": QuantitativeAnalystAgent(),
            "StrategyArchitect": StrategyArchitectAgent(),
            "StrategyRebalancing": StrategyRebalancingAgent(),
            "HedgingStrategist": HedgingStrategistAgent(),
            "StrategyBacktester": StrategyBacktesterAgent(),
            "RegimeForecastingAgent": RegimeForecastingAgent(),
            "FinancialTutor": FinancialTutorAgent(),
        }
        print(f"FinancialOrchestrator initialized with a team of {len(self.roster)} agents.")
        self._setup_classification_patterns()

    def _setup_classification_patterns(self):
        """
        ### IMPROVEMENT: More comprehensive and distinct keyword sets.
        """
        self.routing_map = {
            "StrategyRebalancing": {frozenset(["rebalance"]), frozenset(["optimize"]), frozenset(["allocation"]), frozenset(["risk", "parity"]), frozenset(["diversify", "risk"])},
            "StrategyBacktester": {frozenset(["backtest"]), frozenset(["test", "strategy"])},
            "HedgingStrategist": {frozenset(["hedge"]), frozenset(["protect"]), frozenset(["target", "volatility"])},
            "StrategyArchitect": {frozenset(["design"]), frozenset(["find", "strategy"]), frozenset(["recommend"])},
            "FinancialTutor": {frozenset(["explain"]), frozenset(["what", "is"]), frozenset(["define"]), frozenset(["learn"])},
            "RegimeForecastingAgent": {frozenset(["forecast"]), frozenset(["regime", "change"]), frozenset(["transition"])},
            "QuantitativeAnalyst": {
                frozenset(["risk"]), frozenset(["report"]), frozenset(["factor"]), 
                frozenset(["alpha"]), frozenset(["cvar"]), frozenset(["analyze"]),
                frozenset(["stress", "test"]), frozenset(["tail", "risk"])
            }
        }

    def _classify_query(self, query_words: Set[str]) -> Optional[str]:
        """
        Classifies the query and returns the name of the best agent to handle it.
        This version prioritizes the order defined in the routing_map.
        """
        # ### IMPROVEMENT: The order of this loop now matters for tie-breaking ###
        # We check for the most specific, action-oriented agents first.
        for agent_name, keyword_sets in self.routing_map.items():
            for keyword_set in keyword_sets:
                if keyword_set.issubset(query_words):
                    # Return the first agent that has a match.
                    return agent_name
        return None # No specific agent found

    def route_query(self, user_query: str, db_session, current_user) -> Dict[str, Any]:
        """
        The main entry point. It now uses the improved classification.
        """
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
    
if __name__ == '__main__':
    orchestrator = FinancialOrchestrator()
    
    test_query_1 = "rebalance my portfolio to diversify risk"
    test_query_2 = "What investment strategy do you recommend?"
    
    print("\n--- TESTING ROUTING ---")
    
    # This should now select StrategyRebalancing
    clean_q1 = re.sub(r'[^\w\s]', '', test_query_1.lower())
    words_q1 = set(clean_q1.split())
    agent_for_q1 = orchestrator._classify_query(words_q1)
    print(f"Query: '{test_query_1}' -> Routed to: {agent_for_q1}")

    # This should now select StrategyArchitect
    clean_q2 = re.sub(r'[^\w\s]', '', test_query_2.lower())
    words_q2 = set(clean_q2.split())
    agent_for_q2 = orchestrator._classify_query(words_q2)
    print(f"Query: '{test_query_2}' -> Routed to: {agent_for_q2}")