# in tools/nlp_tools.py
import json
from typing import Dict, Any
from langchain.tools import tool
import anthropic

# ### THE CRITICAL IMPORT: Get settings from our central config ###
from core.config import settings

@tool("SummarizeAnalysisResults")
def summarize_analysis_results(analysis_type: str, data: Dict[str, Any]) -> str:
    """
    Uses Claude 3.5 Sonnet to generate a human-readable, narrative summary
    of a structured JSON analysis result.
    """
    # ### THE FIX: Get the API key from our central settings object, NOT os.getenv ###
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not found in your configuration."
    
    client = anthropic.Anthropic(api_key=api_key)

    data_json_string = json.dumps(data, indent=2)
    
    system_prompt = (
        "You are an expert financial analyst serving as the 'Market Narrator' for an AI platform named Gertie.ai. "
        "Your task is to write a concise, professional, and insightful summary of a financial analysis report. "
        "Do not just list the numbers; provide a narrative interpretation."
    )
    
    user_prompt = f"""
    Here is the report for you to summarize.

    Analysis Type: {analysis_type}

    JSON Data:
    {data_json_string}

    Based on the data above, write a fluid, one-paragraph summary for the user. 
    Focus on the most important takeaways. For a risk report, highlight the overall risk-adjusted return (Sharpe Ratio) and the potential downside (Max Drawdown or CVaR).
    """

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=512,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        summary = message.content[0].text
        return summary if summary else "No summary could be generated."
    except Exception as e:
        print(f"Claude API call failed: {e}")
        return "An error occurred while generating the summary."