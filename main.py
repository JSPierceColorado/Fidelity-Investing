import os
import time
import math
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
import pandas_ta as ta
from twilio.rest import Client as TwilioClient

from alpaca.data.historical import (
    StockHistoricalDataClient,
    CryptoHistoricalDataClient,
)
from alpaca.data.requests import (
    StockBarsRequest,
    CryptoBarsRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

# ───────────────────────────
# Environment / Config
# ───────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() in {"1", "true", "yes"}

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_FROM = os.getenv("TWILIO_FROM")   # e.g. +12025550123
ALERT_TO    = os.getenv("ALERT_TO")       # e.g. +13035550123

# Assets to monitor
STOCK_ETFS: List[str] = ["VIG", "BND", "GLD"]  # VNQ dropped
CRYPTO_PAIRS: List[str] = ["BTC/USD"]          # Alpaca crypto symbol format

# 15-minute bars
TIMEFRAME = TimeFrame(15, TimeFrameUnit.Minute)

# Need at least 240 bars for long SMA; fetch a cushion
LOOKBACK_BARS = 280

# ───────────────────────────
# Helpers
# ───────────────────────────
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def next_quarter_hour(dt: datetime) -> datetime:
    """Return the next 15-minute boundary + 5 seconds (to let bars finalize)."""
    minutes = ((dt.minute // 15) + 1) * 15
    carry = minutes // 60
    next_dt = dt.replace(minute=minutes % 60, second=5, microsecond=0)
    if carry:
        next_dt = next_dt.replace(hour=(dt.hour + 1) % 24)
        if dt.hour == 23:
            next_dt = next_dt + timedelta(days=1)
    return next_dt

def send_sms(message: str):
    if not (TWILIO_SID and TWILIO_AUTH and TWILIO_FROM and ALERT_TO):
        print("[WARN] Twilio env vars not all set; skipping SMS. Message would be:\n", message)
        return
    client = TwilioClient(TWILIO_SID, TWILIO_AUTH)
    client.messages.create(body=message, from_=TWILIO_FROM, to=ALERT_TO)
    print("[INFO] SMS sent")

def to_df_from_bars_list(bars_list) -> pd.DataFrame:
    """Convert a list of Bar objects to a DataFrame we expect."""
    rows = []
    for b in bars_list:
        rows.append({
            "t": b.timestamp,  # tz-aware UTC
            "o": float(b.open),
            "h": float(b.high),
            "l": float(b.low),
            "c": float(b.close),
            "v": float(getattr(b, "volume", np.nan)) if getattr(b, "volume", None) is not None else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    return df

def to_df_from_response(resp, symbol: str) -> pd.DataFrame:
    """Handle both resp.data (dict->list[Bar]) and resp.df (MultiIndex DataFrame)."""
    if hasattr(resp, "data") and resp.data:
        bars_list = resp.data.get(symbol, [])
        return to_df_from_bars_list(bars_list)
    if hasattr(resp, "df") and isinstance(resp.df, pd.DataFrame) and not resp.df.empty:
        # resp.df index: ('symbol', timestamp) or (timestamp) depending on API
        df = resp.df.copy()
        if "symbol" in df.index.names:
            try:
                df = df.xs(symbol, level="symbol")
            except Exception:
                pass
        df = df.reset_index().rename(columns={
            "timestamp": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"
        })
        keep = ["t", "o", "h", "l", "c", "v"]
        df = df[[c for c in keep if c in df.columns]].sort_values("t").reset_index(drop=True)
        return df
    return pd.DataFrame(columns=["t","o","h","l","c","v"])

def compute_indicators(df: pd.DataFrame):
    # Close series
    c = df["c"].astype(float)
    rsi14 = ta.rsi(c, length=14)
    sma60 = ta.sma(c, length=60)
    sma240 = ta.sma(c, length=240)
    return rsi14.iloc[-1], sma60.iloc[-1], sma240.iloc[-1]

def is_nan_or_inf(x: float) -> bool:
    return (
        x is None
        or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    )

def condition_to_buy(rsi, ma60, ma240) -> bool:
    if any(is_nan_or_inf(v) for v in [rsi, ma60, ma240]):
        return False
    return (rsi <= 30) and (ma60 < ma240)

# ───────────────────────────
# Data clients
# ───────────────────────────
stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
crypto_client = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ───────────────────────────
# Fetchers
# ───────────────────────────
def fetch_stock_bars(symbol: str, end: datetime) -> pd.DataFrame:
    start = end - timedelta(minutes=LOOKBACK_BARS * 15 + 60)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        adjustment=Adjustment.SPLIT,
        feed=DataFeed.IEX if ALPACA_PAPER else DataFeed.SIP,
        limit=LOOKBACK_BARS,
    )
    resp = stock_client.get_stock_bars(req)  # ← FIXED METHOD NAME
    return to_df_from_response(resp, symbol)

def fetch_crypto_bars(pair: str, end: datetime) -> pd.DataFrame:
    start = end - timedelta(minutes=LOOKBACK_BARS * 15 + 60)
    req = CryptoBarsRequest(
        symbol_or_symbols=pair,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        limit=LOOKBACK_BARS,
    )
    resp = crypto_client.get_crypto_bars(req)  # ← FIXED METHOD NAME
    return to_df_from_response(resp, pair)

# ───────────────────────────
# Main evaluation loop
# ───────────────────────────
def evaluate_once() -> List[str]:
    ts = now_utc()
    print(f"[INFO] Evaluating at {ts.isoformat()}")
    candidates: List[str] = []

    # Stocks/ETFs
    for sym in STOCK_ETFS:
        try:
            df = fetch_stock_bars(sym, ts)
            if len(df) < 240:
                print(f"[WARN] {sym}: insufficient bars ({len(df)})")
                continue
            rsi, ma60, ma240 = compute_indicators(df)
            print(f"[DEBUG] {sym} RSI14={rsi:.2f} SMA60={ma60:.4f} SMA240={ma240:.4f}")
            if condition_to_buy(rsi, ma60, ma240):
                candidates.append(sym)
        except Exception as e:
            print(f"[ERROR] {sym} fetch/eval failed: {e}")

    # Crypto
    for pair in CRYPTO_PAIRS:
        try:
            df = fetch_crypto_bars(pair, ts)
            if len(df) < 240:
                print(f"[WARN] {pair}: insufficient bars ({len(df)})")
                continue
            rsi, ma60, ma240 = compute_indicators(df)
            print(f"[DEBUG] {pair} RSI14={rsi:.2f} SMA60={ma60:.4f} SMA240={ma240:.4f}")
            if condition_to_buy(rsi, ma60, ma240):
                # Present nicely in SMS (BTCUSD)
                label = pair.replace("/", "")
                candidates.append(label)
        except Exception as e:
            print(f"[ERROR] {pair} fetch/eval failed: {e}")

    return candidates

def loop_forever():
    while True:
        picks = evaluate_once()
        if picks:
            msg = f"BUY SIGNAL {now_utc().strftime('%Y-%m-%d %H:%M:%SZ')}: " + ", ".join(picks)
            print("[ALERT] ", msg)
            send_sms(msg)
        else:
            print("[INFO] No signals this interval.")

        # Sleep until next 15-minute boundary
        nxt = next_quarter_hour(now_utc())
        sleep_s = max(5, (nxt - now_utc()).total_seconds())
        print(f"[INFO] Sleeping {int(sleep_s)}s until {nxt.isoformat()}")
        time.sleep(sleep_s)

# ───────────────────────────
# Entrypoint
# ───────────────────────────
if __name__ == "__main__":
    required = [
        ("ALPACA_API_KEY", ALPACA_API_KEY),
        ("ALPACA_SECRET_KEY", ALPACA_SECRET_KEY),
    ]
    missing = [k for k, v in required if not v]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    loop_forever()
