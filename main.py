import os


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
label = pair.replace("/", "") # BTCUSD
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
