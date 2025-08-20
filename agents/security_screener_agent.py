# agents/security_screener_agent.py
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import requests 

# --- NEW/MODIFIED IMPORTS ---
from io import StringIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# ----------------------------

from agents.base_agent import BaseFinancialAgent, DebatePerspective

class SecurityScreenerAgent(BaseFinancialAgent):
    # __init__ is unchanged
    def __init__(self):
        super().__init__("security_screener", DebatePerspective.SPECIALIST)
        self.tools = []
        self.tool_map = {}
        self.factor_definitions = {
            'quality': {'metrics': ['roe', 'debt_to_equity'], 'description': 'Financial health and profitability'},
            'value': {'metrics': ['pe_ratio', 'pb_ratio'], 'description': 'Attractive valuation metrics'},
            'growth': {'metrics': ['revenue_growth', 'earnings_growth'], 'description': 'Business expansion potential'}
        }
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.screening_universe = self._get_sp500_tickers()
        self.factor_data_cache = None
        self.cache_expiry = None

    # --- MODIFIED: Fixed the FutureWarning ---
    def _get_sp500_tickers(self) -> List[str]:
        try:
            self.logger.info("Fetching S&P 500 ticker list from Wikipedia...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            # FIX: Wrap the response text in StringIO to address the FutureWarning
            tables = pd.read_html(StringIO(response.text))
            sp500_df = tables[0]
            tickers = sp500_df['Symbol'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            self.logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers.")
            return tickers
        except Exception as e:
            self.logger.error(f"Could not fetch S&P 500 tickers: {e}. Falling back to a default list.")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'JNJ', 'V']

    def _get_or_fetch_factor_data(self) -> pd.DataFrame:
        # This caching logic is unchanged
        now = datetime.now()
        if self.factor_data_cache is not None and self.cache_expiry and now < self.cache_expiry:
            self.logger.info("Loading factor data from cache.")
            return self.factor_data_cache
        self.logger.info("Cache is stale or empty. Fetching fresh factor data...")
        raw_data = self._fetch_financial_data(self.screening_universe)
        calculated_data = self._calculate_factor_scores(raw_data)
        self.factor_data_cache = calculated_data
        self.cache_expiry = now + timedelta(days=1)
        return self.factor_data_cache

    # --- NEW: Helper function for multithreading ---
    def _fetch_single_ticker_data(self, ticker_symbol):
        """Fetches data for one ticker; designed to be run in a separate thread."""
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            return {
                'ticker': ticker_symbol, 'roe': info.get('returnOnEquity'),
                'debtToEquity': info.get('debtToEquity'), 'trailingPE': info.get('trailingPE'),
                'priceToBook': info.get('priceToBook'), 'revenueGrowth': info.get('revenueGrowth'),
                'earningsGrowth': info.get('earningsGrowth'),
            }
        except Exception:
            # self.logger.warning(f"Could not fetch data for {ticker_symbol}. Skipping.")
            return None # Return None on failure

    # --- MODIFIED: Switched to a fast, multithreaded approach ---
    def _fetch_financial_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetches raw financial metrics using multithreading for speed and robustness.
        """
        self.logger.info(f"Fetching financial data for {len(tickers)} tickers using parallel requests...")
        all_data = []
        # Use ThreadPoolExecutor to make requests in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Create a future for each ticker fetch
            future_to_ticker = {executor.submit(self._fetch_single_ticker_data, ticker): ticker for ticker in tickers}
            # Process results as they complete, with a tqdm progress bar
            for future in tqdm(as_completed(future_to_ticker), total=len(tickers), desc="Fetching Data"):
                result = future.result()
                if result: # Only append if the fetch was successful
                    all_data.append(result)
        
        self.logger.info(f"Financial data fetching complete. Successfully retrieved data for {len(all_data)}/{len(tickers)} tickers.")
        return pd.DataFrame(all_data)

    # All methods from here down are unchanged.
    def _calculate_factor_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating factor scores from raw data...")
        df_scored = df.copy().set_index('ticker')
        df_scored.fillna(np.nan, inplace=True)
        roe_rank = df_scored['roe'].rank(ascending=False, pct=True)
        de_rank = df_scored['debtToEquity'].rank(ascending=True, pct=True)
        df_scored['quality_score'] = (roe_rank * 0.6 + de_rank * 0.4).fillna(0.5)
        df_scored['pe_positive'] = df_scored['trailingPE'].apply(lambda x: x if x and x > 0 else np.nan)
        pe_rank = df_scored['pe_positive'].rank(ascending=True, pct=True)
        pb_rank = df_scored['priceToBook'].rank(ascending=True, pct=True)
        df_scored['value_score'] = (pe_rank * 0.5 + pb_rank * 0.5).fillna(0.5)
        rg_rank = df_scored['revenueGrowth'].rank(ascending=False, pct=True)
        eg_rank = df_scored['earningsGrowth'].rank(ascending=False, pct=True)
        df_scored['growth_score'] = (rg_rank * 0.5 + eg_rank * 0.5).fillna(0.5)
        self.logger.info("Factor score calculation complete.")
        return df_scored[['quality_score', 'value_score', 'growth_score']].reset_index()

    @property
    def name(self) -> str: return "SecurityScreener"
    @property
    def purpose(self) -> str: return "Finds specific stocks that complement your portfolio using advanced factor analysis"
    def _get_specialization(self) -> str: return "factor_based_security_screening_and_selection"
    def _get_debate_strengths(self) -> List[str]: return ["factor_analysis", "security_selection", "portfolio_complement_analysis", "fundamental_screening", "quantitative_stock_ranking"]
    def _get_specialized_themes(self) -> Dict[str, List[str]]: return {"screening": ["screen", "find", "select", "identify", "discover"],"factors": ["quality", "value", "growth", "momentum", "factor"],"stocks": ["stocks", "securities", "companies", "equities"],"complement": ["complement", "diversify", "balance", "improve"],"specific": ["specific", "particular", "exact", "concrete", "actionable"]}
    async def _gather_specialized_evidence(self, analysis: Dict, context: Dict) -> List[Dict]:
        evidence = []
        themes = analysis.get("relevant_themes", [])
        if "factors" in themes or "screening" in themes: evidence.append({"type": "factor_analysis", "analysis": "Multi-factor screening identifies securities with superior risk-adjusted characteristics.", "data": "Quality factor shows 2.1% annual alpha, Value factor 1.8% alpha over 10-year period", "confidence": 0.85, "source": "Academic factor research and backtesting"})
        if "complement" in themes: evidence.append({"type": "portfolio_analysis", "analysis": "Factor exposure analysis reveals portfolio tilts and diversification opportunities.", "data": "Current portfolio shows 0.7 quality score, 0.3 value score - value tilt opportunity identified", "confidence": 0.82, "source": "Portfolio factor decomposition analysis"})
        evidence.append({"type": "methodological", "analysis": "Systematic screening reduces behavioral biases in security selection.", "data": "Factor-based selection outperforms discretionary picking by 1.9% annually", "confidence": 0.88, "source": "Quantitative research on systematic vs discretionary selection"})
        return evidence
    async def _generate_stance(self, analysis: Dict, evidence: List[Dict]) -> str:
        themes = analysis.get("relevant_themes", [])
        if "specific" in themes and "stocks" in themes: return "recommend systematic factor-based screening to identify specific securities for portfolio enhancement"
        elif "complement" in themes: return "suggest factor analysis approach to find securities that complement existing portfolio exposures"
        elif "screening" in themes: return "propose comprehensive multi-factor screening of investment universe"
        return "advise factor-based security selection methodology for systematic stock picking"
    async def _identify_general_risks(self, context: Dict) -> List[str]: return ["Factor timing risk and cyclical underperformance", "Data quality and accuracy in factor calculations", "Market regime changes affecting factor efficacy", "Overfitting to historical factor relationships", "Implementation costs and trading friction"]
    async def _identify_specialized_risks(self, analysis: Dict, context: Dict) -> List[str]: return ["Factor crowding from widespread adoption", "Sector concentration in factor-tilted portfolios", "Small universe limiting diversification options", "Fundamental data lag affecting real-time decisions"]
    async def execute_specialized_analysis(self, query: str, context: Dict) -> Dict:
        result = self.run(query, context)
        if result.get("success"):
            result["analysis_type"] = "factor_based_screening"
            result["agent_perspective"] = self.perspective.value
            result["confidence_factors"] = ["Multi-factor model validation", "Portfolio complement analysis", "Risk-adjusted ranking methodology"]
        return result
    async def health_check(self) -> Dict: return {"status": "healthy", "response_time": 0.4, "memory_usage": "normal", "active_jobs": 0, "capabilities": self._get_debate_strengths(), "screening_universe_size": len(self.screening_universe)}

    def run(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        self.logger.info(f"--- {self.name} Agent Received Query: '{user_query}' ---")
        try:
            all_factor_data = self._get_or_fetch_factor_data()
            screening_request = self._parse_screening_request(user_query, context)
            if not screening_request["valid"]: return {"success": False, "error": screening_request["error"]}
            if screening_request["type"] == "complement_portfolio": return self._screen_for_portfolio_complement(context, screening_request, all_factor_data)
            elif screening_request["type"] == "factor_based": return self._screen_by_factors(screening_request, all_factor_data)
            else: return self._general_screening(screening_request, all_factor_data)
        except Exception as e:
            self.logger.error(f"Error in SecurityScreener: {e}", exc_info=True)
            return {"success": False, "error": f"Screening failed: {str(e)}", "agent_used": self.name}
    
    def _parse_screening_request(self, query: str, context: Optional[Dict]) -> Dict:
        query_lower = query.lower()
        if any(word in query_lower for word in ["complement", "diversify", "balance", "improve"]):
            if not context or "holdings_with_values" not in context or not context["holdings_with_values"]: return {"valid": False, "error": "Portfolio complement analysis requires portfolio data."}
            screening_type = "complement_portfolio"
        elif any(word in query_lower for word in self.factor_definitions.keys()): screening_type = "factor_based"
        else: screening_type = "general"
        requested_factors = [f for f in self.factor_definitions if f in query_lower]
        if not requested_factors: requested_factors = list(self.factor_definitions.keys())
        import re
        numbers = re.findall(r'\b(\d+)\b', query)
        num_recommendations = int(numbers[0]) if numbers else 3
        return {"valid": True, "type": screening_type, "factors": requested_factors, "num_recommendations": min(max(num_recommendations, 1), 10)}

    def _analyze_portfolio_factors(self, context: Dict, factor_data: pd.DataFrame) -> Dict:
        try:
            holdings = context.get("holdings_with_values", [])
            total_value = sum(h['market_value'] for h in holdings)
            if total_value == 0: return {"success": False, "error": "Portfolio total value is zero."}
            portfolio_df = pd.DataFrame(holdings)
            portfolio_df['weight'] = portfolio_df['market_value'] / total_value
            merged_df = pd.merge(portfolio_df, factor_data, on='ticker', how='left').fillna(0.5)
            exposure = {
                'quality': (merged_df['quality_score'] * merged_df['weight']).sum(),
                'value': (merged_df['value_score'] * merged_df['weight']).sum(),
                'growth': (merged_df['growth_score'] * merged_df['weight']).sum(),
            }
            return {"success": True, "factor_exposure": exposure}
        except Exception as e:
            self.logger.error(f"Portfolio factor analysis failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _screen_for_portfolio_complement(self, context: Dict, request: Dict, factor_data: pd.DataFrame) -> Dict:
        portfolio_analysis = self._analyze_portfolio_factors(context, factor_data)
        if not portfolio_analysis['success']: return portfolio_analysis
        portfolio_factors = portfolio_analysis['factor_exposure']
        self.logger.info(f"Analyzing portfolio with factor exposures: {portfolio_factors}")
        df = factor_data.copy()
        quality_weight = 1 - portfolio_factors['quality']
        value_weight = 1 - portfolio_factors['value']
        growth_weight = 1 - portfolio_factors['growth']
        df['complement_score'] = (df['quality_score'] * quality_weight + df['value_score'] * value_weight + df['growth_score'] * growth_weight)
        portfolio_tickers = [h['ticker'] for h in context.get("holdings_with_values", [])]
        df = df[~df['ticker'].isin(portfolio_tickers)]
        top_candidates_df = df.sort_values(by='complement_score', ascending=False).head(request["num_recommendations"])
        candidates = []
        for _, row in top_candidates_df.iterrows():
            candidates.append({'ticker': row['ticker'], 'overall_score': row['complement_score'], 'factor_scores': {'quality': row['quality_score'], 'value': row['value_score'], 'growth': row['growth_score']}, 'rationale': "Selected to balance your portfolio's factor tilts."})
        return self._format_screening_results(candidates, "portfolio_complement", portfolio_analysis, request)

    def _screen_by_factors(self, request: Dict, factor_data: pd.DataFrame) -> Dict:
        try:
            df = factor_data.copy()
            requested_score_cols = [f + "_score" for f in request["factors"]]
            df['overall_score'] = df[requested_score_cols].mean(axis=1)
            top_candidates_df = df.sort_values(by='overall_score', ascending=False).head(request["num_recommendations"])
            candidates = []
            for _, row in top_candidates_df.iterrows():
                candidates.append({'ticker': row['ticker'], 'overall_score': row['overall_score'], 'factor_scores': {'quality': row['quality_score'], 'value': row['value_score'], 'growth': row['growth_score']}, 'rationale': f"High score in {' & '.join(request['factors'])} factors."})
            return self._format_screening_results(candidates, "factor_based", None, request)
        except Exception as e:
            self.logger.error(f"Factor-based screening failed: {e}")
            return {"success": False, "error": str(e)}

    def _general_screening(self, request: Dict, factor_data: pd.DataFrame) -> Dict:
        df = factor_data.copy()
        df['overall_score'] = df[['quality_score', 'value_score', 'growth_score']].mean(axis=1)
        top_candidates_df = df.sort_values(by='overall_score', ascending=False).head(request["num_recommendations"])
        candidates = []
        for _, row in top_candidates_df.iterrows():
            candidates.append({'ticker': row['ticker'], 'overall_score': row['overall_score'], 'factor_scores': {'quality': row['quality_score'], 'value': row['value_score'], 'growth': row['growth_score']}, 'rationale': "High combined score across Quality, Value, and Growth."})
        return self._format_screening_results(candidates, "general", None, request)

    def _format_screening_results(self, candidates: List[Dict], screening_type: str, portfolio_analysis: Optional[Dict], request: Dict) -> Dict:
        if not candidates: return {"success": False, "error": "No suitable securities found matching criteria"}
        summary = f"### {screening_type.replace('_', ' ').title()} Results\n\n"
        if screening_type == "portfolio_complement" and portfolio_analysis:
            exposure = portfolio_analysis['factor_exposure']
            summary += f"Your portfolio's current factor exposure is Q: {exposure['quality']:.2f}, V: {exposure['value']:.2f}, G: {exposure['growth']:.2f}. "
            summary += "The following securities have been selected to improve this balance:\n\n"
        elif screening_type == "factor_based" and request.get("factors"):
            summary += f"Top securities ranked by {', '.join(request['factors'])} factors:\n\n"
        else:
             summary += f"Top securities based on a balanced factor score:\n\n"
        for i, candidate in enumerate(candidates, 1):
            summary += f"**{i}. {candidate['ticker']}** (Score: {candidate['overall_score']:.2f})\n"
            summary += f"   - {candidate['rationale']}\n"
            fs = candidate['factor_scores']
            summary += f"   - Scores: Q: {fs['quality']:.2f}, V: {fs['value']:.2f}, G: {fs['growth']:.2f}\n\n"
        summary += "\n**Methodology**: Securities are ranked on a percentile basis (0 to 1) for each factor across the S&P 500."
        return {"success": True, "summary": summary, "recommendations": candidates, "screening_type": screening_type, "methodology": "factor_based_screening", "agent_used": self.name, "factors_analyzed": request.get("factors", ["quality", "value", "growth"]), "universe_screened": len(self.screening_universe)}