from datetime import datetime, timedelta, timezone

import pandas_market_calendars as calendars
import yfinance as yf


EXCHANGE_CALENDARS = {
    "NYQ": "NYSE",
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NSI": "NSE",
    "NSE": "NSE",
    "BOM": "BSE",
    "BSE": "BSE",
}


def daily_prices(ticker: str, period: str, *, now: datetime | None = None):
    """Fetch daily bars and require the latest completed session when known."""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period, interval="1d", auto_adjust=False, timeout=10)
    if history.empty:
        raise ValueError("Yahoo Finance returned no daily prices.")
    history = history.dropna(subset=["Close"]).sort_index()
    if history.empty or history.index.tz is None:
        raise ValueError("Yahoo Finance returned no usable exchange-dated prices.")

    metadata = stock.history_metadata
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(history.index.tz).date()
    calendar_name = EXCHANGE_CALENDARS.get(metadata.get("exchangeName"))
    if calendar_name:
        calendar = calendars.get_calendar(calendar_name)
        schedule = calendar.schedule(today - timedelta(days=14), today)
        completed = schedule.loc[schedule["market_close"] <= now]
        if completed.empty:
            raise ValueError("Could not determine the latest completed trading session.")
        expected = completed.index[-1].date()
        history = history.loc[history.index.date <= expected]
    else:
        # Without a known schedule, only accept yesterday's daily bar.
        expected = today - timedelta(days=1)
        history = history.loc[history.index.date <= expected]

    latest = history.index[-1].date() if not history.empty else None
    if latest != expected:
        observed = latest.isoformat() if latest else "none"
        raise ValueError(f"Fresh daily prices unavailable. Expected {expected}, received {observed}.")

    return history, metadata.get("currency") or "currency unavailable"
