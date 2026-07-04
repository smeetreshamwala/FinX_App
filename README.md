# FinX Smart Suggestion App

`FinX Smart Suggestion App` is a Streamlit-based Python app for stock recommendation and portfolio analysis. It uses Yahoo Finance data through `yfinance`, builds interactive charts with Plotly, calculates technical indicators with Pandas and NumPy, and produces a transparent rule-based `AI Smartness Score` without any external AI API or paid service.

## What the App Does

- Analyzes a single stock ticker such as `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `AAPL`, or `MSFT`
- Supports multiple historical timeframes: `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, and `max`
- Displays the latest close, date range, daily change, and 52-week high/low
- Builds a Plotly closing-price chart with a 50-day moving average
- Calculates pivot point support and resistance levels: Pivot, `R1-R3`, and `S1-S3`
- Calculates RSI (14), MACD (12, 26, 9), and mean reversion z-score
- Fetches available fundamentals including valuation, profitability, growth, leverage, sector, and industry
- Scores the stock from `0-100` using technical, fundamental, and risk-based rules
- Generates a transparent `Buy`, `Hold`, or `Sell` recommendation
- Lets users upload a CSV portfolio and runs live analysis for each holding
- Calculates portfolio summary metrics, weights, volatility, historical VaR, downside risk, and upside estimate
- Suggests portfolio changes using the same rule-based scoring system

## Project Structure

```text
FinX Scripts/
├── suggest.py
├── requirements.txt
└── README.md
```

## AI Smartness Score Logic

The recommendation engine is fully rule-based and does not call OpenAI, Gemini, or any other external AI API.

The final score is built from three blocks:

1. Technical score
   - RSI condition
   - MACD condition
   - Mean reversion z-score condition
   - Price trend versus the 50-day moving average
   - Proximity to support and resistance
2. Fundamental score
   - Trailing P/E
   - Price-to-book ratio
   - Debt-to-equity ratio
   - Profit margins
   - Revenue growth
   - Return on equity
3. Risk score
   - Historical volatility
   - Maximum drawdown
   - Distance from support
   - Distance from resistance

Recommendation rules:

- `Score >= 70`: `Buy`
- `Score between 40 and 69`: `Hold`
- `Score < 40`: `Sell`

The app also shows the key reasons and the score breakdown so the recommendation is easy to audit.

## Technical Indicators Included

### Pivot Levels

Calculated using the latest available OHLC values:

```text
Pivot Point = (High + Low + Close) / 3
Resistance 1 = (2 * Pivot Point) - Low
Support 1 = (2 * Pivot Point) - High
Resistance 2 = Pivot Point + (High - Low)
Support 2 = Pivot Point - (High - Low)
Resistance 3 = High + 2 * (Pivot Point - Low)
Support 3 = Low - 2 * (High - Pivot Point)
```

### RSI

- `RSI > 70`: Overbought
- `RSI < 30`: Oversold
- `RSI 30 to 70`: Neutral

### MACD

- Fast EMA: `12`
- Slow EMA: `26`
- Signal line: `9`
- `MACD above signal`: Bullish
- `MACD below signal`: Bearish
- `MACD close to signal`: Neutral

### Mean Reversion

- Rolling mean: `20`
- Rolling standard deviation: `20`
- Z-score: `(Close - Rolling Mean) / Rolling Std Dev`
- `Z > +2`: Overbought
- `Z < -2`: Oversold
- `-2 to +2`: Neutral

## Portfolio CSV Format

Upload a CSV file with the following columns:

```csv
Ticker,Quantity,Average_Buy_Price
RELIANCE.NS,10,2800
TCS.NS,5,3900
INFY.NS,8,1450
```

Required rules:

- `Ticker` must be a valid Yahoo Finance symbol
- `Quantity` must be numeric and greater than zero
- `Average_Buy_Price` must be numeric and greater than zero

## Portfolio VaR Logic

For uploaded portfolios, the app calculates:

- Daily returns for each stock
- Portfolio weighted daily return
- Annualized portfolio volatility
- `95% historical VaR`
- Expected downside VaR from negative daily returns
- Upside estimate from the upper return percentile

This is a historical return-based estimate and is intended for educational analysis only.

## Run Locally

Open a terminal in the `FinX Scripts` folder and run:

```bash
pip install -r requirements.txt
streamlit run suggest.py
```

Streamlit will usually open the app at:

```text
http://localhost:8501
```

## Deploy to GitHub

1. Create a new GitHub repository.
2. Upload the contents of `FinX Scripts`.
3. Make sure `suggest.py`, `requirements.txt`, and `README.md` are in the repository.
4. Push your code to the `main` branch.

Example commands:

```bash
git init
git add .
git commit -m "Add FinX Smart Suggestion App"
git branch -M main
git remote add origin https://github.com/your-username/your-repository-name.git
git push -u origin main
```

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with your GitHub account.
4. Click `New app`.
5. Select your repository and branch.
6. Set the main file path to `suggest.py`.
7. Click `Deploy`.

Streamlit Community Cloud will install the packages from `requirements.txt` automatically.

## Important Disclaimer

This app is for educational and analytical purposes only.

- It does not provide guaranteed investment advice.
- The `AI Smartness Score` is a transparent rule-based model.
- Yahoo Finance data may be delayed, incomplete, or unavailable for some tickers.
- Always do your own research before making financial decisions.
