# in agents/conversation_manager.py
import os
import autogen
from typing import List, Dict, Any

class AutoGenConversationManager:
    def __init__(self, llm_config: Dict[str, Any]):
        """Initializes the manager with a configuration for the LLM."""
        self.llm_config = llm_config

    def start_conversation(self, user_query: str, portfolio_context: str) -> Dict[str, Any]:
        """
        Starts and manages a multi-agent group chat to solve a complex query.
        NOTE: For this first version, these agents are pure LLMs playing a role.
        The next major upgrade will be to give them access to our Python tools.
        """
        # 1. Define the Agents in the group chat
        user_proxy = autogen.UserProxyAgent(
           name="User_Proxy",
           human_input_mode="NEVER",
           max_consecutive_auto_reply=0,
           code_execution_config=False, # We are not executing code in this version
        )

        quant_analyst = autogen.AssistantAgent(
            name="QuantitativeAnalyst",
            llm_config=self.llm_config,
            system_message="You are a world-class quantitative analyst. Your job is to analyze the user's portfolio, identify key risks (like high volatility, drawdown, or poor Sharpe ratio), and present your findings with specific numbers."
        )

        strategy_architect = autogen.AssistantAgent(
            name="StrategyArchitect",
            llm_config=self.llm_config,
            system_message="You are a world-class investment strategist. Your job is to listen to the QuantitativeAnalyst's findings and propose a new, improved target portfolio allocation that addresses the identified risks and aligns with common investment goals."
        )

        rebalancer = autogen.AssistantAgent(
            name="StrategyRebalancingAgent",
            llm_config=self.llm_config,
            system_message="You are a pragmatic portfolio manager. Your job is to take the new target allocation from the StrategyArchitect and describe the high-level trades needed to get there (e.g., 'Sell tech stocks, buy bonds')."
        )

        planner = autogen.AssistantAgent(
            name="Planner",
            llm_config=self.llm_config,
            system_message="You are the lead financial advisor and planner. Your job is to review the entire conversation between the other specialists. Synthesize their findings into a single, cohesive, three-part final report for the user. The report MUST have three sections: '1. Risk Assessment', '2. New Strategic Allocation', and '3. Actionable Plan'. After presenting the complete plan, you MUST end your message with the single word: TERMINATE"
        )

        # 2. Create the Group Chat
        groupchat = autogen.GroupChat(
            agents=[user_proxy, quant_analyst, strategy_architect, rebalancer, planner],
            messages=[],
            max_round=12
        )

        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config,
            is_termination_msg=lambda x: x.get("name") == "Planner" and "TERMINATE" in x.get("content", "")
        )

        # 3. Initiate the Chat
        initial_prompt = f"""
        The user has a complex request: "{user_query}"
        Here is their current portfolio context: {portfolio_context}

        Please work together as a team to formulate a complete, actionable plan. Follow this sequence:
        1. The QuantitativeAnalyst starts by assessing the current portfolio's risks.
        2. The StrategyArchitect proposes a new target allocation based on the analyst's findings.
        3. The StrategyRebalancingAgent describes the trades needed.
        4. The Planner synthesizes everything into a final report and terminates the chat.
        Begin.
        """
        user_proxy.initiate_chat(manager, message=initial_prompt)

        # 4. Extract the final response
        final_response = "The AI team has completed their analysis." # Fallback
        for msg in reversed(groupchat.messages):
            if msg.get('name') == 'Planner' and msg.get('content', ''):
                final_response = msg['content'].replace("TERMINATE", "").strip()
                break

        return {"summary": final_response, "full_conversation": groupchat.messages}