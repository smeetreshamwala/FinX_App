# FinX Smart Suggestion App

`FinX Smart Suggestion App` remains fully inside `FinX Scripts/Suggest/` and follows the project rule that each future script must live inside its own subfolder under `FinX Scripts`.

## Folder Structure

```text
FinX Scripts/
└── Suggest/
    ├── suggest.py
    ├── requirements.txt
    └── README.md
```

## Full Feature List

- Single-stock analysis with an NSE dropdown universe using `.NS` normalization
- Portfolio upload and live portfolio intelligence with CSV ticker normalization
- Pivot support and resistance, RSI, MACD, mean reversion, and candlestick pattern detection
- Fundamental analysis from `yfinance`
- Expanded rule-based `AI Smartness Score` with dynamic component weighting
- Buy / Hold / Sell recommendations with bullish reasons, bearish risks, and missing-data notes
- Multi-stock screener for `Nifty 50`, `F&O Stocks`, `S&P 500`, `NASDAQ`, and custom watchlists
- Relative strength ranking with weighted 20/50/100/200-day momentum
- Options chain analysis with PCR, max pain, IV skew, and top open-interest strikes
- Volatility forecasting with `GARCH` when available and a rolling-volatility fallback otherwise
- Forecasting with `ARIMA` and optional experimental `LSTM`
- Portfolio optimization using `Markowitz`, inverse-volatility HRP approximation, and simplified Black-Litterman
- Monte Carlo portfolio simulation
- Kelly criterion position sizing
- Sector and industry concentration analysis
- FII / DII market activity dashboard with manual entry or CSV upload
- Demo login and local JSON-based saved watchlists
- Streamlit caching and a manual refresh button

## Local File Rule

Every future script should use its own subfolder inside `FinX Scripts`, for example:

```text
script_name_folder/
├── main_script.py
├── requirements.txt
└── README.md
```

## Ticker Handling

- Indian symbols are normalized with `.NS`
- Portfolio uploads accept `RELIANCE` and `RELIANCE.NS`
- Screener universes use `.NS` only for Indian symbols
- US ticker universes use standard Yahoo Finance symbols without `.NS`

## Portfolio CSV Format

```csv
Ticker,Quantity,Average_Buy_Price
RELIANCE,10,2800
TCS.NS,5,3900
INFY,8,1450
```

Required columns:

- `Ticker`
- `Quantity`
- `Average_Buy_Price`

## FII / DII CSV Format

```csv
Date,FII_Net,DII_Net
2026-07-03,-1200,850
2026-07-04,900,300
```

## AI Smartness Score

The app does not use OpenAI, Gemini, Claude, or any external AI API. All recommendations remain rule-based.

Target weights:

- `Technical Score`: `30%`
- `Fundamental Score`: `20%`
- `Relative Strength Score`: `15%`
- `Risk Score`: `15%`
- `Options Sentiment Score`: `10%`
- `Institutional Activity Score`: `10%`

If options or FII/DII data is unavailable, the app redistributes the weights across available components automatically.

Recommendation bands:

- `Score >= 70`: `Buy`
- `Score 40 to 69`: `Hold`
- `Score < 40`: `Sell`

## Relative Strength Ranking

The screener calculates weighted momentum:

```text
0.40 * 20-day return
+ 0.30 * 50-day return
+ 0.20 * 100-day return
+ 0.10 * 200-day return
```

Missing 100-day or 200-day data is handled safely and the score uses available lookbacks only.

## Options Chain Metrics

The options tab uses Yahoo Finance data when available and shows:

- Calls and puts tables
- Put-call ratio by open interest
- Put-call ratio by volume
- Max pain
- Implied-volatility skew
- Highest call open-interest strikes
- Highest put open-interest strikes

If options data is missing for a ticker, the app shows a warning and continues safely.

## Volatility Forecasting

- Uses `arch` for `GARCH(1,1)` forecasting when the optional package is available
- Falls back to 20-day rolling annualized volatility when `arch` is unavailable or the fit fails
- This fallback is displayed clearly in the UI

`arch` is optional and is not required in the default `requirements.txt`.

## ARIMA and LSTM Forecasting

- `ARIMA` is the default forecasting model via `statsmodels`
- Forecast horizons: `5`, `10`, or `20` trading days
- `LSTM` is optional and experimental
- `TensorFlow` is not included in the default requirements because it can be heavy on Streamlit Community Cloud
- If `TensorFlow` is missing, the app warns the user and continues with ARIMA

## Portfolio Optimization

The optimization tab supports:

- `Markowitz Mean-Variance`
- `Hierarchical Risk Parity approximation` via inverse-volatility weighting
- `Black-Litterman simplified approximation` using user views layered onto historical returns

It also includes:

- Current vs optimized allocation
- Allocation difference
- Expected return
- Expected volatility
- Sharpe ratio
- Monte Carlo simulation
- Kelly sizing estimates

## Demo Authentication and Watchlists

- Demo login only
- Username: `demo`
- Password: `demo123`
- Uses `st.session_state` plus a local `watchlists.json` file
- This is not production-grade authentication and should not be used for sensitive data

## Run Locally

```bash
cd "FinX Scripts/Suggest"
pip install -r requirements.txt
streamlit run suggest.py
```

Optional extras for advanced modules:

- `arch` for true GARCH forecasts
- `tensorflow` for the experimental LSTM forecast

## Deploy to GitHub

1. Create a GitHub repository.
2. Add the `FinX Scripts` folder with the `Suggest` subfolder.
3. Commit and push to the `main` branch.

Example:

```bash
git init
git add .
git commit -m "Add FinX Suggest dashboard"
git branch -M main
git remote add origin https://github.com/your-username/your-repository-name.git
git push -u origin main
```

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app.
4. Choose the repository and branch.
5. Set the main file path to `FinX Scripts/Suggest/suggest.py` if the repository root includes `FinX Scripts`.
6. Deploy.

Notes for Cloud deployment:

- `TensorFlow` is intentionally omitted from the default requirements
- `arch` is optional
- Large screener runs are intentionally capped for performance

## Financial Disclaimer

Disclaimer: This application is for educational and analytical purposes only. It does not provide guaranteed investment advice, trading signals, or financial recommendations. All Buy / Hold / Sell outputs are generated using transparent rule-based scoring and should be independently verified before making investment decisions.
