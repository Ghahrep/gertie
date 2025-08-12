# in tools/behavioral_tools.py
from typing import Dict, Any, List
from langchain.tools import tool

@tool("AnalyzeChatForBiases")
def analyze_chat_for_biases(chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyzes a user's chat history to detect potential behavioral biases.
    Looks for patterns like frequent switching, chasing performance, or fear-based language.
    """
    biases_found = {}
    user_messages = [msg['content'].lower() for msg in chat_history if msg.get('role') == 'user']
    
    # 1. Loss Aversion / Panic Selling Detection
    loss_aversion_keywords = ['panic', 'sell everything', 'market crash', 'get out', 'afraid of losing']
    if any(any(kw in msg for kw in loss_aversion_keywords) for msg in user_messages):
        biases_found['Loss Aversion'] = {
            "finding": "Detected language related to panic or selling based on fear.",
            "suggestion": "Consider sticking to your long-term plan. Emotional decisions during downturns can often hurt performance."
        }

    # 2. Herding / FOMO Detection
    herding_keywords = ['everyone is buying', 'hot stock', 'get in on', 'don\'t want to miss out', 'fomo']
    if any(any(kw in msg for kw in herding_keywords) for msg in user_messages):
        biases_found['Herding Behavior (FOMO)'] = {
            "finding": "Detected language suggesting a desire to follow the crowd or chase popular trends.",
            "suggestion": "Ensure investment decisions are based on your own research and strategy, not just popularity."
        }

    # 3. Overconfidence / Frequent Rebalancing Detection
    rebalance_queries = [msg for msg in user_messages if 'rebalance' in msg]
    if len(rebalance_queries) > 2: # Simple check for frequent rebalancing requests
        biases_found['Over-trading / Overconfidence'] = {
            "finding": "Noticed multiple requests for rebalancing in a short period.",
            "suggestion": "Frequent trading can increase costs and may not always lead to better results. Ensure each change aligns with your long-term goals."
        }
        
    if not biases_found:
        return {"success": True, "summary": "No strong behavioral biases were detected in the recent conversation."}

    return {"success": True, "biases_detected": biases_found}