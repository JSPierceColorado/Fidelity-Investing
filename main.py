import os
import time
from datetime import datetime, timedelta, timezone
import math
import pandas as pd
import numpy as np
import pandas_ta as ta
from twilio.rest import Client as TwilioClient

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() in {"1", "true", "yes"}

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_FROM = os.getenv("TWILIO_FROM")   # e.g. +12025550123
ALERT_TO    = os.getenv("ALERT_TO")       # e.g. +13035550123

# Assets to monitor
STOCK_ETFS = ["VIG", "BND", "GLD"]
CRYPTO_PAIRS = ["BTC/USD"]  # Alpaca crypto format

TIMEFRAME = TimeFrame.Minute * 15  # 15-minute bars
# Need at least 240 bars for the long SMA; fetch a cushion
LOOKBACK_BARS = 280

# ---- Helpers ----

def now_utc():
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

def to_df(bars) -> pd.DataFrame:
    rows = []
    for b in bars:
        rows.append({
            "t": b.timestamp,  # already timezone-aware UTC
            "o": float(b.open),
            "h": float(b.high),
            "l": float(b.low),
            "c": float(b.close),
            "v": float(b.volume) if hasattr(b, 'volume') and b.volume is not None else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    return df

def compute_indicators(df: pd.DataFrame):
    # Close series
    c = df["c"].astype(float)
    rsi14 = ta.rsi(c, length=14)
    sma60 = ta.sma(c, length=60)
    sma240 = ta.sma(c, length=240)
    return rsi14.iloc[-1], sma60.iloc[-1], sma240.iloc[-1]

def condition_to_buy(rsi, ma60, ma240):
    if any(map(lambda x: x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))), [rsi, ma60, ma240])):
        return False
    return (rsi <= 30) and (ma60 < ma240)

# ---- Data Fetchers ----

stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
crypto_client = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def fetch_stock_bars(symbol: str, end: datetime) -> pd.DataFrame:
    start = end - timedelta(minutes=LOOKBACK_BARS * 15 + 60)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        adjustment="split",
        feed="sip" if not ALPACA_PAPER else "iex",  # paper often limited to IEX
        limit=LOOKBACK_BARS,
    )
    resp = stock_client.get_bars(req)
    bars = resp[symbol]
    return to_df(bars)

def fetch_crypto_bars(pair: str, end: datetime) -> pd.DataFrame:
    start = end - timedelta(minutes=LOOKBACK_BARS * 15 + 60)
    req = CryptoBarsRequest(
        symbol_or_symbols=pair,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        limit=LOOKBACK_BARS,
    )
    resp = crypto_client.get_bars(req)
    bars = resp[pair]
    return to_df(bars)

# ---- Main loop ----

def evaluate_once() -> list[str]:
    ts = now_utc()
    print(f"[INFO] Evaluating at {ts.isoformat()}")
    candidates = []

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
                # Present as BTC for the SMS label
                label = pair.replace("/", "")  # BTCUSD
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

if __name__ == "__main__":
    required = [
        ("ALPACA_API_KEY", ALPACA_API_KEY),
        ("ALPACA_SECRET_KEY", ALPACA_SECRET_KEY),
    ]
    missing = [k for k, v in required if not v]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    loop_forever()
