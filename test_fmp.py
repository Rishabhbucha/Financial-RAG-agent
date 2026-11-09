import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("FMP_API_KEY")
if not key:
    raise SystemExit("FMP_API_KEY is missing.")

r = requests.get(
    "https://financialmodelingprep.com/stable/income-statement",
    params={"symbol": "AAPL", "period": "annual", "limit": 1, "apikey": key},
    timeout=10,
)
print(r.status_code)
print(r.text[:200])
