# in agents/base_agent.py
from abc import ABC, abstractmethod
from typing import List, Any, Dict,Optional

class BaseFinancialAgent(ABC):
    """
    An abstract base class for all specialized financial AI agents.
    It enforces a standard interface for initialization and execution.
    """
    def __init__(self, tools: List[Any]):
        self.tools = tools
        # A simple way to map tool names to the tool objects for easy access
        self.tool_map = {tool.name: tool for tool in tools}

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the agent."""
        pass

    @property
    @abstractmethod
    def purpose(self) -> str:
        """A brief description of what this agent does."""
        pass
    
    @abstractmethod
    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        The main entry point for the agent to process a query.
        
        Parameters:
        -----------
        user_query : str
            The natural language query from the user.
        context : Optional[Dict]
            Additional context, like portfolio data or conversation history.
            
        Returns:
        --------
        Dict[str, Any]
            A structured dictionary containing the agent's findings.
        """
        pass