import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("FMP_API_KEY")
print(f"Key used: {key}")

url = f"https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?timeseries=30&apikey={key}"
r = requests.get(url)
print(r.status_code)
print(r.text[:200])
