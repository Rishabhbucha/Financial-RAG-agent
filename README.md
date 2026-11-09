# Financial-RAG-agent

FinRAG is an agentic AI financial research workspace built with Python and LangGraph. It combines local RAG (ChromaDB) for PDFs/Images with live tools: DuckDuckGo web search, FMP API for real-time stock data, and dynamic charting. Powered by a self-critiquing ReAct agent, it grades and refines its own answers for maximum accuracy.

## Features
- **Multimodal Document Ingestion:** Upload PDFs, DOCX, and Images. Text is extracted, chunked, and stored locally in a ChromaDB vector database.
- **Daily Market Data:** Fetches dated closing prices and volume from Yahoo Finance using yfinance. Quotes and charts require the latest completed session for NYSE, Nasdaq, NSE, and BSE, accounting for weekends and scheduled holidays. For other exchanges, the tool requires yesterday's bar in the exchange's timezone. If the required data is missing, the tool reports it as unavailable.
- **Company Fundamentals:** Uses FMP Stable endpoints for company profiles and annual income statements. Statements follow company reporting schedules and show their period end and filing date when available.
- **Dynamic Charting:** Saves historical closing-price line charts using Matplotlib and attempts to open them on Windows.
- **Live Web Search:** Built-in DuckDuckGo search ensures the AI isn't limited by its training data cutoff and can fetch real-time news.
- **Self-Critiquing Agent:** Built using LangGraph, the agent uses a ReAct loop to generate a draft, critique its own accuracy/faithfulness, and automatically rewrite its answer if it doesn't meet quality standards.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Rishabhbucha/Financial-RAG-agent.git
cd Financial-RAG-agent
```

### 2. Install dependencies
Install all the required Python libraries using pip:
```bash
pip install -r requirements.txt
```

### 3. Set up Environment Variables
Create a file named `.env` in the root directory. Copy the following into the file and replace the placeholders with your actual API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
FMP_API_KEY=your_financial_modeling_prep_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```
*Note: If you don't have a Tavily API key, leave it blank. The agent will automatically fall back to the free DuckDuckGo search.*

`FMP_API_KEY` is needed for company profiles and income statements. Quotes and charts use Yahoo Finance symbols, such as `AAPL` and `RELIANCE.NS`.

## Usage

To start the agent, simply run the main script:

```bash
python main.py
```

1. **Document Upload:** The script will first ask if you want to upload a financial document. If you do, provide the absolute path to your file (e.g., `C:\Documents\report.pdf`). The system will parse it and add it to the local vector database.
2. **Chatting:** Once setup is complete, you will drop into an interactive chat prompt. You can ask Fin things like:
   - *"Compare the revenue in my uploaded report with Apple's latest financial performance."*
   - *"What is the latest news regarding TSLA?"*
   - *"Create a 6-month historical price chart for Microsoft (MSFT)."*

The agent will autonomously decide which tools to use, gather the data, critique its own findings, and provide you with a final, highly accurate answer.
