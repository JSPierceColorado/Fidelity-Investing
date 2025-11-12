# RSI/MA Alert Bot (Alpaca → Telegram)

A lean Python bot that watches **15‑minute bars** for a small list of assets (default: **VIG, BND, GLD, BTC**) using **Alpaca Market Data** (stocks/ETFs + crypto) and sends minimal **Telegram** alerts when oversold + bearish‑trend conditions occur.

> Example alert: `buy VIG` or graded: `strong buy BTCUSD`

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-stable-brightgreen" />
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-blue" />
  <img alt="alerts" src="https://img.shields.io/badge/alerts-Telegram-26A5E4" />
  <img alt="data" src="https://img.shields.io/badge/data-Alpaca-000000" />
</p>

---

## ✨ What it does

* Pulls **15m bars** from Alpaca for a curated watchlist (stocks/ETFs and BTC).
* Computes **RSI(14)** and moving averages to detect oversold + downtrend setups.
* **Stocks/ETFs rule:** `RSI14 < 30` **and** `SMA60 < SMA240`.
* **BTC rule:** `RSI14 < 30` **and** `SMA180 < SMA720`.
* Optionally computes **% down from ATH** (daily bars) and upgrades messages to `weak/moderate/strong buy`.
* Sends short **Telegram** messages; throttles to **one alert per asset per UTC day**.
* Runs forever; aligns work to the next **15‑minute boundary**.

---

## 📦 Requirements

* Python **3.11+**
* Alpaca **Market Data** API key/secret
* A **Telegram Bot** and target chat/channel ID

Install deps:

```bash
pip install -r requirements.txt
```

Minimal packages if you roll your own:

```
numpy
pandas
pandas-ta
requests
alpaca-py
python-dateutil
```

---

## 🔧 Configuration (env vars)

| Variable             | Required | Example                           | Notes                                                         |
| -------------------- | :------: | --------------------------------- | ------------------------------------------------------------- |
| `ALPACA_API_KEY`     |     ✅    | `PK...`                           | Alpaca API key                                                |
| `ALPACA_SECRET_KEY`  |     ✅    | `...`                             | Alpaca secret                                                 |
| `ALPACA_PAPER`       |          | `true`                            | Chooses IEX feed for paper (`true`) or SIP for live (`false`) |
| `TELEGRAM_BOT_TOKEN` |          | `123456:ABC...`                   | Needed to send messages                                       |
| `TELEGRAM_CHAT_ID`   |          | `123456789` or `-100xxxxxxxxxxxx` | User/Group/Channel target                                     |
| `SEND_BOOT_TELEGRAM` |          | `true`                            | Sends a startup notice                                        |
| `ALERT_LOG_PATH`     |          | `/mnt/data/alert_log.json`        | Persists last‑alert date per asset                            |

> **Note:** `TELEGRAM_*` vars are optional. If missing, messages are printed to stdout instead of sent.

### Optional watchlist

Edit in code:

```python
STOCK_ETFS = ["VIG", "BND", "GLD"]
CRYPTO_PAIRS = ["BTC/USD"]
```

---

## ▶️ Running

### Local

```bash
export ALPACA_API_KEY=... \
       ALPACA_SECRET_KEY=... \
       ALPACA_PAPER=true \
       TELEGRAM_BOT_TOKEN=123456:ABC... \
       TELEGRAM_CHAT_ID=-1001234567890 \
       SEND_BOOT_TELEGRAM=true

python main.py
```

### Docker (example)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py"]
```

```bash
docker build -t rsi-ma-alert-bot .
docker run --rm \
  -e ALPACA_API_KEY -e ALPACA_SECRET_KEY -e ALPACA_PAPER=true \
  -e TELEGRAM_BOT_TOKEN -e TELEGRAM_CHAT_ID -e SEND_BOOT_TELEGRAM=true \
  -e ALERT_LOG_PATH=/mnt/data/alert_log.json \
  -v $(pwd)/data:/mnt/data \
  rsi-ma-alert-bot
```

### GitHub Actions (optional schedule)

Create `.github/workflows/alerts.yml`:

```yaml
name: RSI/MA Alert Bot

on:
  workflow_dispatch: {}
  schedule:
    # Every 15 min (UTC). Adjust if needed.
    - cron: "*/15 * * * *"

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run
        env:
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
          ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
          ALPACA_PAPER: ${{ secrets.ALPACA_PAPER }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          SEND_BOOT_TELEGRAM: ${{ secrets.SEND_BOOT_TELEGRAM }}
          ALERT_LOG_PATH: /tmp/alert_log.json
        run: |
          python main.py
```

> GitHub cron runs in **UTC**. The bot itself already aligns to 15‑minute boundaries.

Badge (after first run):

```md
![build](https://github.com/<you>/<repo>/actions/workflows/alerts.yml/badge.svg)
```

---

## 🧠 Strategy & signals

**Stocks/ETFs:**

* Compute RSI(14), SMA60, SMA240 on 15m closes
* **Alert** when: `RSI14 < 30` **and** `SMA60 < SMA240`

**BTC:**

* Compute RSI(14), SMA180, SMA720 on 15m closes
* **Alert** when: `RSI14 < 30` **and** `SMA180 < SMA720`

**Grading (optional):**

* Fetch daily history to estimate **ATH** and current close; compute `% down from ATH`.
* Map to buckets: ≤10% `weak buy`; 10–25% `moderate buy`; ≥50% `strong buy`; else `buy`.

**Deduplication:**

* One alert per **asset per UTC day** (persisted in `ALERT_LOG_PATH`).

---

## 🔒 Security

* Treat Alpaca and Telegram tokens as **secrets** (`.env`, GitHub Secrets, Docker/K8s secret stores).
* Do **not** commit credentials. Mount env vars or secret files at runtime.

---

## 🛠 Troubleshooting

* **Missing required env vars** → the program exits with a clear message.
* **No Telegram sent** → verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; otherwise messages print to stdout.
* **Few/zero bars** → check market hours for ETFs, or that crypto has enough lookback (`LOOKBACK_BARS_CRYPTO`).
* **Too many alerts** → that shouldn’t happen: alerts are daily‑deduped; check `ALERT_LOG_PATH` persistence.

---

## 🤝 Contributing

PRs and issues welcome. Keep alerts minimal by design; if adding context, guard it behind a flag.

## 📜 License

MIT — see `LICENSE`.
