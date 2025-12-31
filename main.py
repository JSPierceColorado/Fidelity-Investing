"""
RSI/MA Alert Bot (15m bars): VIG, BND, GLD, BTC
- Uses Alpaca Market Data (stocks/ETFs + crypto) for indicators
- Sends alerts via Telegram (free)
- Buy alerts are minimal: "buy <asset>" (no timestamp)
- Optional boot notification controlled by SEND_BOOT_TELEGRAM

Env vars:
  # Alpaca
  ALPACA_API_KEY=...
  ALPACA_SECRET_KEY=...
  ALPACA_PAPER=true|false

  # Telegram
  TELEGRAM_BOT_TOKEN=123456:ABC...
  TELEGRAM_CHAT_ID=123456789           # or -100xxxxxxxxxxxx for groups/channels
  SEND_BOOT_TELEGRAM=true|false

  # Optional (where to persist last-alert dates)
  ALERT_LOG_PATH=/mnt/data/alert_log.json
"""

import os
import time
import math
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

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

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SEND_BOOT_TELEGRAM = os.getenv("SEND_BOOT_TELEGRAM", "false").lower() in {"1", "true", "yes"}

# Where we persist per-asset last-alert dates (UTC YYYY-MM-DD)
ALERT_LOG_PATH = Path(os.getenv("ALERT_LOG_PATH", "/tmp/alert_log.json"))

# Assets to monitor (VNQ removed)
STOCK_ETFS: List[str] = ["VIG", "BND", "GLD", "USDU"]
CRYPTO_PAIRS: List[str] = ["BTC/USD"]  # Alpaca crypto symbol format

# 15-minute bars
TIMEFRAME = TimeFrame(15, TimeFrameUnit.Minute)

# Need at least 240 bars for long SMA (stocks) and 720 bars for BTC crypto rule
LOOKBACK_DAYS_STOCK = 30       # calendar days (market hours only)
LOOKBACK_BARS_CRYPTO = 900     # ↑ ensure >=720 bars + cushion for BTC rule

# Daily timeframe for ATH checks
TIMEFRAME_DAILY = TimeFrame(1, TimeFrameUnit.Day)

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

def send_telegram(text: str):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        print("[WARN] Telegram env vars not set; skipping Telegram. Would send:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    if r.ok:
        print("[INFO] Telegram sent")
    else:
        print(f"[ERROR] Telegram failed {r.status_code} {r.text}")

def to_df_from_bars_list(bars_list) -> pd.DataFrame:
    """Convert a list of Bar objects to a DataFrame."""
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

# ── Indicators ─────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> Tuple[float, float, float]:
    """RSI14, SMA60, SMA240 — used for stocks/ETFs."""
    c = df["c"].astype(float)
    rsi14 = ta.rsi(c, length=14)
    sma60 = ta.sma(c, length=60)
    sma240 = ta.sma(c, length=240)
    return float(rsi14.iloc[-1]), float(sma60.iloc[-1]), float(sma240.iloc[-1])

def compute_indicators_ma(df: pd.DataFrame, short_len: int, long_len: int) -> Tuple[float, float, float]:
    """RSI14, SMA(short_len), SMA(long_len) — generic (used for BTC rule)."""
    c = df["c"].astype(float)
    rsi14 = ta.rsi(c, length=14)
    sma_s = ta.sma(c, length=short_len)
    sma_l = ta.sma(c, length=long_len)
    return float(rsi14.iloc[-1]), float(sma_s.iloc[-1]), float(sma_l.iloc[-1])

def is_nan_or_inf(x: float) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

def condition_to_buy(rsi, ma60, ma240) -> bool:
    """Stocks/ETFs rule: RSI14 < 30 and SMA60 < SMA240."""
    if any(is_nan_or_inf(v) for v in [rsi, ma60, ma240]):
        return False
    return (rsi < 30) and (ma60 < ma240)

def condition_to_buy_btc(rsi, ma180, ma720) -> bool:
    """BTC rule: RSI14 < 30 and SMA180 < SMA720."""
    if any(is_nan_or_inf(v) for v in [rsi, ma180, ma720]):
        return False
    return (rsi < 30) and (ma180 < ma720)

# ── Alert deduplication (one alert per asset per UTC day) ──────────
def utc_date_str(dt: datetime) -> str:
    """Return YYYY-MM-DD for a tz-aware UTC datetime."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")

def load_alert_log() -> dict:
    """Load per-asset last-alert UTC dates from disk."""
    try:
        if ALERT_LOG_PATH.exists():
            with ALERT_LOG_PATH.open("r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        print(f"[WARN] Failed to load alert log: {e}")
    return {}

def save_alert_log(log: dict):
    """Persist the alert log to disk."""
    try:
        ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ALERT_LOG_PATH.open("w") as f:
            json.dump(log, f)
    except Exception as e:
        print(f"[WARN] Failed to save alert log: {e}")

def should_alert(asset: str, ts: datetime, log: dict) -> bool:
    """True if we have not sent an alert for `asset` on the UTC day of `ts`."""
    today = utc_date_str(ts)
    last = log.get(asset)
    return last != today

def mark_alert(asset: str, ts: datetime, log: dict):
    """Record that we sent an alert for `asset` on the UTC day of `ts`."""
    log[asset] = utc_date_str(ts)
    save_alert_log(log)

# Initialize alert log after helpers are defined
ALERT_LOG = load_alert_log()

# Cache of last computed %down-from-ATH per asset label (e.g. 'VIG', 'BTCUSD')
LAST_PCT_DOWN: dict = {}

# ───────────────────────────
# Data clients
# ───────────────────────────
stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
crypto_client = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ───────────────────────────
# ATH helpers (minimal-impact add)
# ───────────────────────────
def is_crypto_label(asset: str) -> bool:
    """Detect alert labels like 'BTCUSD' (or raw pairs containing '/')."""
    return asset.upper().endswith("USD") or ("/" in asset)

def to_pair_from_label(asset: str) -> str:
    """Convert 'BTCUSD' -> 'BTC/USD' (no-op if already a pair)."""
    return f"{asset[:-3]}/{asset[-3:]}" if "/" not in asset else asset

def fetch_daily_ath_and_last(symbol_or_pair: str, is_crypto: bool) -> Tuple[float, float]:
    """
    Return (ath_high, last_close). Raises on hard failures.
    Uses deep daily history to approximate "all-time" within provider coverage.
    """
    start_stocks = datetime(2000, 1, 1, tzinfo=timezone.utc)
    start_crypto = datetime(2015, 1, 1, tzinfo=timezone.utc)
    end = now_utc()

    if is_crypto:
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol_or_pair,
            timeframe=TIMEFRAME_DAILY,
            start=start_crypto,
            end=end,
            limit=100000,
        )
        resp = crypto_client.get_crypto_bars(req)
        df = to_df_from_response(resp, symbol_or_pair)
    else:
        req = StockBarsRequest(
            symbol_or_symbols=symbol_or_pair,
            timeframe=TIMEFRAME_DAILY,
            start=start_stocks,
            end=end,
            adjustment=Adjustment.SPLIT,
            feed=DataFeed.IEX if ALPACA_PAPER else DataFeed.SIP,
            limit=100000,
        )
        resp = stock_client.get_stock_bars(req)
        df = to_df_from_response(resp, symbol_or_pair)

    if df.empty:
        raise RuntimeError("No daily bars")

    ath = float(df["h"].max())
    last_close = float(df["c"].iloc[-1])
    if ath <= 0 or last_close <= 0:
        raise RuntimeError("Nonpositive prices encountered")
    return ath, last_close

def compute_pct_down_from_ath(asset_label: str) -> float:
    """
    asset_label is like 'VIG' or 'BTCUSD' (your existing alert label).
    Returns % down from ATH as a positive float (e.g., 23.4),
    or raises if it cannot compute.
    """
    if is_crypto_label(asset_label):
        pair = to_pair_from_label(asset_label)  # 'BTCUSD' -> 'BTC/USD'
        ath, last_close = fetch_daily_ath_and_last(pair, is_crypto=True)
    else:
        ath, last_close = fetch_daily_ath_and_last(asset_label, is_crypto=False)

    pct_down = max(0.0, (ath - last_close) / ath * 100.0)
    return pct_down

def grade_buy(pct_down: float) -> str:
    """
    Buckets:
      - ≤10%      → 'weak buy'
      - 11–25%    → 'moderate buy'
      - ≥26%      → 'strong buy'
    """
    if pct_down <= 10.0:
        return "weak buy"
    elif pct_down <= 25.0:
        return "moderate buy"
    else:
        return "strong buy"

# ───────────────────────────
# Fetchers
# ───────────────────────────
def fetch_stock_bars(symbol: str, end: datetime) -> pd.DataFrame:
    # Fetch ~30 calendar days to accumulate >=240 15m bars (market hours only)
    start = end - timedelta(days=LOOKBACK_DAYS_STOCK)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        adjustment=Adjustment.SPLIT,
        feed=DataFeed.IEX if ALPACA_PAPER else DataFeed.SIP,
        limit=10000,  # high cap to avoid truncation
    )
    resp = stock_client.get_stock_bars(req)
    return to_df_from_response(resp, symbol)

def fetch_crypto_bars(pair: str, end: datetime) -> pd.DataFrame:
    # Crypto trades 24/7; bar count is straightforward.
    # Need >=720 bars for BTC rule; request with cushion.
    start = end - timedelta(minutes=LOOKBACK_BARS_CRYPTO * 15 + 60)
    req = CryptoBarsRequest(
        symbol_or_symbols=pair,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        limit=LOOKBACK_BARS_CRYPTO,
    )
    resp = crypto_client.get_crypto_bars(req)
    return to_df_from_response(resp, pair)

# ───────────────────────────
# Main evaluation loop
# ───────────────────────────
def evaluate_once() -> List[str]:
    ts = now_utc()
    print(f"[INFO] Evaluating at {ts.isoformat()}")
    candidates: List[str] = []

    # Stocks/ETFs (RSI<30 & SMA60<SMA240)
    for sym in STOCK_ETFS:
        try:
            df = fetch_stock_bars(sym, ts)
            if len(df) < 240:
                print(f"[WARN] {sym}: insufficient bars ({len(df)})")
                continue
            rsi, ma60, ma240 = compute_indicators(df)

            # Log %downATH for visibility and cache it
            try:
                pct_down = compute_pct_down_from_ath(sym)
                LAST_PCT_DOWN[sym] = pct_down
            except Exception as e:
                pct_down = float('nan')
                LAST_PCT_DOWN[sym] = pct_down
                print(f"[WARN] {sym}: failed to get %downATH ({e})")

            print(f"[DEBUG] {sym} RSI14={rsi:.2f} SMA60={ma60:.4f} SMA240={ma240:.4f} %downATH={pct_down:.1f}")

            if condition_to_buy(rsi, ma60, ma240):
                candidates.append(sym)
        except Exception as e:
            print(f"[ERROR] {sym} fetch/eval failed: {e}")

    # Crypto (BTC rule: RSI<30 & SMA180<SMA720)
    for pair in CRYPTO_PAIRS:
        try:
            df = fetch_crypto_bars(pair, ts)
            if len(df) < 720:
                print(f"[WARN] {pair}: insufficient bars for BTC rule ({len(df)})")
                continue
            rsi, ma180, ma720 = compute_indicators_ma(df, 180, 720)

            # Log %downATH for visibility (use BTCUSD label for grading/fetch) and cache it
            label = pair.replace("/", "")  # BTC/USD -> BTCUSD
            try:
                pct_down = compute_pct_down_from_ath(label)
                LAST_PCT_DOWN[label] = pct_down
            except Exception as e:
                pct_down = float('nan')
                LAST_PCT_DOWN[label] = pct_down
                print(f"[WARN] {pair}: failed to get %downATH ({e})")

            print(f"[DEBUG] {pair} RSI14={rsi:.2f} SMA180={ma180:.4f} SMA720={ma720:.4f} %downATH={pct_down:.1f}")

            if condition_to_buy_btc(rsi, ma180, ma720):
                candidates.append(label)
        except Exception as e:
            print(f"[ERROR] {pair} fetch/eval failed: {e}")

    return candidates

def loop_forever():
    while True:
        picks = evaluate_once()
        ts_now = now_utc()

        if picks:
            for asset in picks:
                if should_alert(asset, ts_now, ALERT_LOG):
                    # Upgrade message with %downATH-based grading; fall back to minimal 'buy'
                    try:
                        pct = LAST_PCT_DOWN.get(asset)
                        if pct is None or is_nan_or_inf(pct):
                            pct = compute_pct_down_from_ath(asset)
                            LAST_PCT_DOWN[asset] = pct

                        label = grade_buy(pct)
                        msg = f"{label} {asset}"  # keep minimal; add pct if desired
                        # e.g., msg = f"{label} {asset} ({pct:.1f}% down from ATH)"
                    except Exception as e:
                        print(f"[WARN] grading failed for {asset}: {e}")
                        msg = f"buy {asset}"

                    print("[ALERT]", msg)
                    send_telegram(msg)
                    mark_alert(asset, ts_now, ALERT_LOG)
                else:
                    print(f"[INFO] Skipping duplicate alert for {asset} on {utc_date_str(ts_now)}")
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

    if SEND_BOOT_TELEGRAM:
        send_telegram(
            f"✅ Bot started at {now_utc().strftime('%Y-%m-%d %H:%M:%SZ')} "
            f"(15m; ETFs: {', '.join(STOCK_ETFS)}; Crypto: {', '.join([p.replace('/', '') for p in CRYPTO_PAIRS])})"
        )

    loop_forever()
