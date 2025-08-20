# test_quant_risk.py
import logging
from agents.quantitative_analyst import QuantitativeAnalystAgent

# Configure logging to see the agent's output
logging.basicConfig(level=logging.INFO)

def run_risk_test():
    """
    Initializes the QuantitativeAnalystAgent and tests its new
    _calculate_tail_dependency method.
    """
    print("\n--- Testing QuantitativeAnalystAgent's Tail Risk Logic ---")
    
    try:
        quant_agent = QuantitativeAnalystAgent()
        print("✅ Agent initialized successfully.")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return

    # --- Test Case 1: High Correlation (Market vs Tech) ---
    print("\n--- Test Case 1: SPY vs. QQQ ---")
    print("[EXPECTATION] High crash risk, as they are highly correlated market indices.")
    spy_qqq_dependence = quant_agent._calculate_tail_dependency(['SPY', 'QQQ'])
    print(f"==> Result: SPY-QQQ Crash Correlation = {spy_qqq_dependence:.2%}")

    # --- Test Case 2: Potential Hedge (Market vs Gold) ---
    print("\n--- Test Case 2: SPY vs. GLD ---")
    print("[EXPECTATION] Low crash risk, as gold is often a safe-haven asset.")
    spy_gld_dependence = quant_agent._calculate_tail_dependency(['SPY', 'GLD'])
    print(f"==> Result: SPY-GLD Crash Correlation = {spy_gld_dependence:.2%}")
    
    # --- Test Case 3: Competing Assets (Visa vs Mastercard) ---
    print("\n--- Test Case 3: V vs. MA ---")
    print("[EXPECTATION] Very high crash risk, as they are in the exact same industry.")
    v_ma_dependence = quant_agent._calculate_tail_dependency(['V', 'MA'])
    print(f"==> Result: V-MA Crash Correlation = {v_ma_dependence:.2%}")

    print("\n✅ Risk calculation test completed.")

if __name__ == "__main__":
    run_risk_test()