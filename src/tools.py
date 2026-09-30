import os
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

# We will import VectorDBManager here to use it in the tool
# Note: For a more robust app, you might inject this instead of global instantiation.
from .vector_db import VectorDBManager

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
    Retrieve current stock information (price, volume, market cap) for a given ticker symbol (e.g., AAPL).
    """
    if not FMP_API_KEY:
        return "Error: FMP API Key is missing."
        
    url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            quote = data[0]
            return (f"Ticker: {quote['symbol']}\n"
                    f"Price: ${quote['price']}\n"
                    f"Change: {quote['changesPercentage']}%\n"
                    f"Market Cap: {quote['marketCap']}\n"
                    f"Volume: {quote['volume']}")
        return f"No quote data found for {ticker}."
    return f"Failed to fetch stock quote. Status code: {response.status_code}"

@tool
def get_company_profile(ticker: str) -> str:
    """
    Retrieve company profile information (industry, sector, description, CEO) for a given ticker symbol.
    """
    if not FMP_API_KEY:
        return "Error: FMP API Key is missing."
        
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    
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
        
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=1&apikey={FMP_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            stmt = data[0]
            return (f"Year: {stmt['calendarYear']}\n"
                    f"Revenue: ${stmt['revenue']}\n"
                    f"Gross Profit: ${stmt['grossProfit']}\n"
                    f"Operating Income: ${stmt['operatingIncome']}\n"
                    f"Net Income: ${stmt['netIncome']}")
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
    if not FMP_API_KEY:
        return "Error: FMP API Key is missing."
        
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries={days}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "historical" in data and data["historical"]:
            historical = data["historical"]
            # FMP returns newest first; reverse for chronological plotting
            historical.reverse()
            
            dates = [item["date"] for item in historical]
            prices = [item["close"] for item in historical]
            
            plt.figure(figsize=(10, 5))
            plt.plot(dates, prices, marker='o', linestyle='-', color='b')
            plt.title(f"{ticker.upper()} Stock Price - Last {days} Days")
            plt.xlabel("Date")
            plt.ylabel("Closing Price ($)")
            
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
            try:
                os.startfile(filename)
            except Exception:
                pass
                
            return f"Successfully generated a chart for {ticker}. The chart was saved and opened at '{filename}'. Tell the user to look at the popped-up window!"
        return f"No historical data found for {ticker}."
    return f"Failed to fetch historical data. Status code: {response.status_code}"

# List of all tools to be bound to the LangGraph agent
finrag_tools = [
    search_financial_documents,
    get_stock_quote,
    get_company_profile,
    get_financial_statements,
    search_financial_news,
    plot_stock_history
]
