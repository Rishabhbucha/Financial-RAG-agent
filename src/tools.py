import os
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

# We will import VectorDBManager here to use it in the tool
# Note: For a more robust app, you might inject this instead of global instantiation.
from .vector_db import VectorDBManager
from .market_data import daily_prices

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

try:
    vector_db = VectorDBManager()
except Exception as e:
    vector_db = None
    print(f"Warning: VectorDB could not be initialized in tools.py: {e}")

try:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
except Exception:
    tavily_client = None

@tool
def search_financial_documents(query: str) -> str:
    """
    Search uploaded financial documents (like annual reports, statements) for specific information.
    Use this to extract facts from the user's uploaded documents.
    """
    if not vector_db:
        return "Error: Document database is not initialized."
    
    results = vector_db.search(query, n_results=3)
    if not results:
        return "No relevant information found in the uploaded documents."
    
    formatted_results = []
    for r in results:
        source = r['metadata'].get('source', 'Unknown')
        formatted_results.append(f"Source: {source}\nExcerpt: {r['content']}")
        
    return "\n\n---\n\n".join(formatted_results)

@tool
def get_stock_quote(ticker: str) -> str:
    """
    Retrieve the latest verified daily closing price and volume for a Yahoo Finance ticker (e.g., AAPL, RELIANCE.NS).
    """
    ticker = ticker.strip().upper()
    try:
        history, currency = daily_prices(ticker, "1mo")
    except Exception as e:
        return f"Daily market data unavailable for {ticker}: {e}"

    latest = history.iloc[-1]
    closes = history["Close"]
    change = f"{closes.pct_change().iloc[-1]:+.2%}" if len(closes) > 1 else "unavailable"
    return (
        f"Ticker: {ticker}\n"
        f"Source: Yahoo Finance\n"
        f"Trading session: {history.index[-1].date()}\n"
        f"Daily close: {latest['Close']:,.4f} {currency}\n"
        f"Change from previous available close: {change}\n"
        f"Volume: {latest['Volume']:,.0f}"
    )

@tool
def get_company_profile(ticker: str) -> str:
    """
    Retrieve company profile information (industry, sector, description, CEO) for a given ticker symbol.
    """
    if not FMP_API_KEY:
        return "Error: FMP API Key is missing."
        
    response = requests.get(
        "https://financialmodelingprep.com/stable/profile",
        params={"symbol": ticker, "apikey": FMP_API_KEY},
        timeout=10,
    )
    
    if response.status_code == 200:
        data = response.json()
        if data:
            profile = data[0]
            return (f"Company: {profile['companyName']}\n"
                    f"Sector: {profile['sector']}\n"
                    f"Industry: {profile['industry']}\n"
                    f"CEO: {profile['ceo']}\n"
                    f"Description: {profile['description']}")
        return f"No profile found for {ticker}."
    return f"Failed to fetch company profile."

@tool
def get_financial_statements(ticker: str) -> str:
    """
    Retrieve fundamental financial information (revenue, net income, gross profit, operating income) 
    from the latest income statement for a given ticker symbol.
    """
    if not FMP_API_KEY:
        return "Error: FMP API Key is missing."
        
    response = requests.get(
        "https://financialmodelingprep.com/stable/income-statement",
        params={"symbol": ticker, "period": "annual", "limit": 1, "apikey": FMP_API_KEY},
        timeout=10,
    )
    
    if response.status_code == 200:
        data = response.json()
        if data:
            stmt = data[0]
            return (f"Source: Financial Modeling Prep\n"
                    f"Annual period end: {stmt.get('date', 'unavailable')}\n"
                    f"Filed on: {stmt.get('filingDate') or 'unavailable'}\n"
                    f"Reporting currency: {stmt.get('reportedCurrency') or 'unavailable'}\n"
                    f"Revenue: {stmt['revenue']}\n"
                    f"Gross Profit: {stmt['grossProfit']}\n"
                    f"Operating Income: {stmt['operatingIncome']}\n"
                    f"Net Income: {stmt['netIncome']}")
        return f"No financial statements found for {ticker}."
    return "Failed to fetch financial statements."

@tool
def search_financial_news(query: str) -> str:
    """
    Search the web for current financial news, company developments, and market commentary.
    """
    if tavily_client:
        try:
            response = tavily_client.search(query=query, search_depth="basic", include_answer=True)
            # We can combine the generated answer with snippets
            result_text = f"Tavily Answer: {response.get('answer', 'No summary available.')}\n\nSources:\n"
            for result in response.get('results', [])[:3]:
                result_text += f"- {result['title']}: {result['url']}\n  {result['content']}\n\n"
            return result_text
        except Exception as e:
            return f"Web search failed: {e}"
    else:
        # 100% Free Fallback: DuckDuckGo
        try:
            from langchain_community.tools import DuckDuckGoSearchResults
            ddg = DuckDuckGoSearchResults()
            return f"DuckDuckGo Search Results:\n{ddg.invoke(query)}"
        except Exception as e:
            return f"DuckDuckGo search failed: {e}"

@tool
def plot_stock_history(ticker: str, days: int = 30) -> str:
    """
    Fetches historical stock prices for a given ticker and generates a line chart.
    Use this when the user asks to see a chart, graph, visual trend, or historical performance of a stock.
    """
    try:
        ticker = ticker.strip().upper()
        if days < 1:
            return "The chart requires at least one trading session."
        hist, currency = daily_prices(ticker, "1y")
            
        hist = hist.tail(days)
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        
        plt.figure(figsize=(10, 5))
        plt.plot(dates, prices, marker='o', linestyle='-', color='b')
        plt.title(f"{ticker} Closing Price - {len(hist)} Sessions Through {dates[-1]}")
        plt.xlabel("Date")
        plt.ylabel(f"Closing Price ({currency})")
        
        # Show only a few date labels to avoid crowding
        if len(dates) > 10:
            step = len(dates) // 10
            plt.xticks(ticks=range(0, len(dates), step), labels=[dates[i] for i in range(0, len(dates), step)], rotation=45)
        else:
            plt.xticks(rotation=45)
            
        plt.grid(True)
        plt.tight_layout()
        
        os.makedirs("charts", exist_ok=True)
        filename = os.path.abspath(f"charts/{ticker}_history.png")
        plt.savefig(filename)
        plt.close()
        
        # Automatically open the chart for the user on Windows!
        status = ""
        try:
            os.startfile(filename)
            status = " Opened the saved chart."
        except Exception:
            pass
            
        return f"Yahoo Finance chart for {ticker} through {dates[-1]} saved at '{filename}'.{status}"
    except Exception as e:
        return f"Failed to fetch historical data: {str(e)}"

# List of all tools to be bound to the LangGraph agent
finrag_tools = [
    search_financial_documents,
    get_stock_quote,
    get_company_profile,
    get_financial_statements,
    search_financial_news,
    plot_stock_history
]
