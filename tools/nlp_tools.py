import json
from typing import Dict, Any
from langchain.tools import tool
import anthropic

# Import settings from our central config
from core.config import settings

@tool("SummarizeAnalysisResults")
def summarize_analysis_results(analysis_type: str, data: Dict[str, Any]) -> str:
    """
    Uses Claude 3.5 Sonnet to generate a human-readable, narrative summary
    of a structured JSON analysis result.
    """
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        print("ERROR in SummarizeAnalysisResults: ANTHROPIC_API_KEY not found in settings.")
        return "Error: API key for the summarization service is not configured."
    
    try:
        client = anthropic.Anthropic(api_key=api_key)

        data_json_string = json.dumps(data, indent=2)
        
        system_prompt = (
            "You are an expert financial analyst. Your task is to write a concise, professional, and "
            "insightful summary of a financial analysis report. Do not just list the numbers; "
            "provide a narrative interpretation."
        )
        
        user_prompt = f"""
        Here is the report for you to summarize.

        Analysis Type: {analysis_type}

        JSON Data:
        {data_json_string}

        Based on the data above, write a fluid, one-paragraph summary for the user. 
        Focus on the most important takeaways. For a risk report, highlight the overall risk-adjusted return (Sharpe Ratio) and the potential downside (Max Drawdown or CVaR).
        """

        print("--- Calling Anthropic API for summary... ---")
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=512,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        summary = message.content[0].text
        print("--- Successfully received summary from Anthropic API. ---")
        return summary if summary else "No summary could be generated."

    except anthropic.APIError as e:
        # --- THIS IS THE CRITICAL ADDITION ---
        # This will catch specific errors from the Anthropic API (like auth errors)
        # and print detailed information.
        print("!!! ANTHROPIC API ERROR !!!")
        print(f"Error Type: {type(e)}")
        print(f"Status Code: {e.status_code}")
        print(f"Response Body: {e.response.text}")
        print("!!! END ANTHROPIC API ERROR !!!")
        return f"An error occurred while generating the summary: API call failed with status {e.status_code}."
        
    except Exception as e:
        # This is a general catch-all for other unexpected errors.
        print(f"An unexpected error occurred in SummarizeAnalysisResults: {e}")
        return "An unexpected error occurred while generating the summary."

