from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="FinX Smart Suggestion App",
    page_icon="📊",
    layout="wide",
)


PERIOD_OPTIONS = {
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
    "Maximum available data": "max",
}

FORECAST_HORIZONS = [5, 10, 20]
REQUIRED_PORTFOLIO_COLUMNS = ["Ticker", "Quantity", "Average_Buy_Price"]
REQUIRED_FII_DII_COLUMNS = ["Date", "FII_Net", "DII_Net"]
WATCHLIST_FILE = Path(__file__).with_name("watchlists.json")
DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demo123"
APP_DISCLAIMER = (
    "Disclaimer: This application is for educational and analytical purposes only. "
    "It does not provide guaranteed investment advice, trading signals, or financial "
    "recommendations. All Buy / Hold / Sell outputs are generated using transparent "
    "rule-based scoring and should be independently verified before making investment decisions."
)


RAW_STOCKS = [
    "RELIANCE",
    "HDFCBANK",
    "BHARTIARTL",
    "ICICIBANK",
    "SBIN",
    "TCS",
    "TVSMOTOR",
    "BAJFINANCE",
    "LT",
    "HINDUNILVR",
    "LICI",
    "INFY",
    "SUNPHARMA",
    "ADANIPOWER",
    "MARUTI",
    "ADANIPORTS",
    "AXISBANK",
    "M&M",
    "ITC",
    "KOTAKBANK",
    "NTPC",
    "ONGC",
    "TITAN",
    "ADANIENT",
    "ULTRACEMCO",
    "M&MFIN",
    "HCLTECH",
    "JSWSTEEL",
    "BEL",
    "BAJAJ-AUTO",
    "HAL",
    "BAJAJFINSV",
    "COALINDIA",
    "POWERGRID",
    "NESTLEIND",
    "DMART",
    "HINDZINC",
    "TATASTEEL",
    "ASIANPAINT",
    "HINDALCO",
    "ETERNAL",
    "ADANIGREEN",
    "SHRIRAMFIN",
    "WIPRO",
    "GRASIM",
    "IOC",
    "EICHERMOT",
    "SBILIFE",
    "DIVISLAB",
    "VBL",
    "BSE",
    "INDIGO",
    "SOLARINDS",
    "ADANIENSOL",
    "POWERINDIA",
    "ICICIAMC",
    "JIOFIN",
    "TORNTPHARM",
    "CUMMINSIND",
    "PIDILITIND",
    "IDEA",
    "TRENT",
    "HYUNDAI",
    "DLF",
    "BHEL",
    "PFC",
    "TECHM",
    "ABB",
    "TMCV",
    "MOTHERSON",
    "POLYCAB",
    "BANKBARODA",
    "CGPOWER",
    "TMPV",
    "MUTHOOTFIN",
    "TATAPOWER",
    "SIEMENS",
    "HDFCLIFE",
    "VEDL",
    "CHOLAFIN",
    "BRITANNIA",
    "TATACAP",
    "BPCL",
    "IRFC",
    "JINDALSTEL",
    "ENRIN",
    "LTM",
    "UNIONBANK",
    "GVT&D",
    "TATACONSUM",
    "GROWW",
    "PNB",
    "BAJAJHLDNG",
    "APOLLOHOSP",
    "CANBK",
    "HDFCAMC",
    "CIPLA",
    "INDUSTOWER",
    "DRREDDY",
    "INDIANB",
    "MARICO",
    "BOSCHLTD",
    "AMBUJACEM",
    "MANKIND",
    "ZYDUSLIFE",
    "MAXHEALTH",
    "GODREJCP",
    "LUPIN",
    "LGEINDIA",
    "GAIL",
    "GMRAIRPORT",
    "HEROMOTOCO",
    "MAZDOCK",
    "JSWENERGY",
    "INDHOTEL",
    "UNITDSPR",
    "ABCAPITAL",
    "BHARATFORG",
    "ASHOKLEY",
    "LLOYDSME",
    "NTPCGREEN",
    "ICICIGI",
    "MEESHO",
    "SHREECEM",
    "LODHA",
    "MCX",
    "RECLTD",
    "AUROPHARMA",
    "WAAREEENER",
    "POLICYBZR",
    "LENSKART",
    "OFSS",
    "SAIL",
    "OIL",
    "HINDPETRO",
    "PERSISTENT",
    "DABUR",
    "NHPC",
    "BHARTIHEXA",
    "IDBI",
    "NYKAA",
    "NMDC",
    "SRF",
    "TORNTPOWER",
    "HAVELLS",
    "NATIONALUM",
    "ICICIPRULI",
    "PAYTM",
    "FORTIS",
    "LAURUSLABS",
    "AUBANK",
    "SUZLON",
    "FEDERALBNK",
    "SWIGGY",
    "NAM-INDIA",
    "BIOCON",
    "INDUSINDBK",
    "LTF",
    "YESBANK",
    "BAJAJHFL",
    "GICRE",
    "ATGL",
    "GLENMARK",
    "DIXON",
    "ALKEM",
    "IOB",
    "BANKINDIA",
    "SCHAEFFLER",
    "PHOENIXLTD",
    "LINDEINDIA",
    "NAUKRI",
    "UNOMINDA",
    "JSL",
    "MAHABANK",
    "PRESTIGE",
    "COLPAL",
    "SBICARD",
    "BERGEPAINT",
    "OBEROIRLTY",
    "IDFCFIRSTB",
    "ABBOTINDIA",
    "TIINDIA",
    "VMM",
    "RVNL",
    "FACT",
    "MFSL",
    "HDBFS",
    "COROMANDEL",
    "JSWINFRA",
    "MRF",
    "HINDCOPPER",
    "UPL",
    "THERMAX",
    "GODREJPROP",
    "APLAPOLLO",
    "MOTILALOFS",
    "APARINDS",
    "PATANJALI",
    "TATACOMM",
    "SUNDARMFIN",
    "NLCINDIA",
    "KEI",
    "BDL",
    "RADICO",
    "COFORGE",
    "360ONE",
    "PREMIERENE",
    "SUPREMEIND",
    "PIIND",
    "ANTHEM",
    "PIRAMALFIN",
    "AIIL",
    "VOLTAS",
    "PAGEIND",
    "MPHASIS",
    "IRCTC",
    "IPCALAB",
    "BALKRISIND",
    "JKCEMENT",
    "HUDCO",
    "FLUOROCHEM",
    "AJANTPHARM",
    "GLAXO",
    "PETRONET",
    "COCHINSHIP",
    "ASTERDM",
    "NH",
    "ASTRAL",
    "CONCOR",
    "LTTS",
    "GODREJIND",
    "AIAENG",
    "GLAND",
    "SONACOMS",
    "NAVINFLUOR",
    "GODFRYPHLP",
    "ENDURANCE",
    "UBL",
    "IREDA",
    "KALYANKJIL",
    "POONAWALLA",
    "3MINDIA",
    "ATHERENERG",
    "JBCHEPHARM",
    "WELCORP",
    "BLUESTARCO",
    "DELHIVERY",
    "TATAINVEST",
    "MEDANTA",
    "EMCURE",
    "KPRMILL",
    "ESCORTS",
    "PWL",
    "DALBHARAT",
    "ITCHOTELS",
    "HEXT",
    "JUBLFOOD",
    "SCHNEIDER",
    "BANDHANBNK",
    "CRISIL",
    "UCOBANK",
    "CENTRALBK",
    "CHOLAHLDNG",
    "STARHEALTH",
    "GRSE",
    "KIMS",
    "LICHSGFIN",
    "MANAPPURAM",
    "ANGELONE",
    "ABSLAMC",
    "ANANDRATHI",
    "HSCL",
    "SJVN",
    "EXIDEIND",
    "GODIGIT",
    "CPPLUS",
    "LALPATHLAB",
    "IKS",
    "KARURVYSYA",
    "PNBHOUSING",
    "TIMKEN",
    "ITI",
    "TATATECH",
    "ZFCVINDIA",
    "NUVAMA",
    "NIACL",
    "HONAUT",
    "MSUMI",
    "WOCKPHARMA",
    "TATAELXSI",
    "MRPL",
    "SHYAMMETL",
    "CDSL",
    "AWL",
    "IRB",
    "ACC",
    "NBCC",
    "GILLETTE",
    "AEGISLOG",
    "DEEPAKNTR",
    "FORCEMOT",
    "AMBER",
    "SAILIFE",
    "KIRLOSENG",
    "TENNIND",
    "CESC",
    "APOLLOTYRE",
    "GESHIP",
    "PTCIL",
    "ACUTAAS",
    "PPLPHARMA",
    "SUMICHEM",
    "RRKABEL",
    "AEGISVOPAK",
    "NETWEB",
    "IGL",
    "BAYERCROP",
    "DATAPATTNS",
    "PFIZER",
    "ONESOURCE",
    "NEULANDLAB",
    "HFCL",
    "HBLENGINE",
    "KPIL",
    "ASAHIINDIA",
    "AADHARHFC",
    "TRITURBINE",
    "RAMCOCEM",
    "CREDITACC",
    "NATCOPHARM",
    "ATUL",
    "KPITTECH",
    "GPIL",
    "CRAFTSMAN",
    "GMDCLTD",
    "SUNTV",
    "AFFLE",
    "RBLBANK",
    "ERIS",
    "CARBORUNIV",
    "EIHOTEL",
    "KAYNES",
    "IIFL",
    "SAGILITY",
    "SARDAEN",
    "SYRMA",
    "CAMS",
    "GRANULES",
    "SYNGENE",
    "EMAMILTD",
    "CROMPTON",
    "BELRISE",
    "TATACHEM",
    "CUB",
    "URBANCO",
    "EMMVEE",
    "CASTROLIND",
    "ELGIEQUIP",
    "CGCL",
    "CHAMBLFERT",
    "NAVA",
    "VTL",
    "ANANTRAJ",
    "CIEINDIA",
    "REDINGTON",
    "CHALET",
    "FSL",
    "PINELABS",
    "DCMSHRIRAM",
    "ACMESOLAR",
    "BIKAJI",
    "AARTIIND",
    "KAJARIACER",
    "DEEPAKFERT",
    "BRIGADE",
    "IFCI",
    "JSWCEMENT",
    "INOXWIND",
    "SAMMAANCAP",
    "ZYDUSWELL",
    "OLAELEC",
    "GALLANTT",
    "JUBLPHARMA",
    "COHANCE",
    "FINCABLES",
    "GABRIEL",
    "ANURAS",
    "CAPLIPOINT",
    "SCI",
    "JYOTICNC",
    "BEML",
    "IGIL",
    "JBMA",
    "CHENNPETRO",
    "NIVABUPA",
    "ARE&M",
    "ABDL",
    "CHOICEIN",
    "POLYMED",
    "J&KBANK",
    "SOBHA",
    "CCL",
    "CEMPRO",
    "ZENTEC",
    "GRAPHITE",
    "DEVYANI",
    "KFINTECH",
    "ABREL",
    "ECLERX",
    "EIDPARRY",
    "USHAMART",
    "JINDALSAW",
    "LTFOODS",
    "TECHNOE",
    "TRAVELFOOD",
    "DOMS",
    "RAINBOW",
    "THELEELA",
    "WELSPUNLIV",
    "ENGINERSIN",
    "JAINREC",
    "APTUS",
    "JSWDULUX",
    "IRCON",
    "PGEL",
    "CANHLIFE",
    "FIVESTAR",
    "VIJAYA",
    "SPLPETRO",
    "PARADEEP",
    "CEATLTD",
    "TBOTEK",
    "NSLNISP",
    "KEC",
    "JPPOWER",
    "INDIACEM",
    "ABLBL",
    "INDGN",
    "MINDACORP",
    "TRIDENT",
    "JMFINANCIL",
    "UTIAMC",
    "INDIAMART",
    "GRAVITA",
    "TEGA",
    "JUBLINGREA",
    "CONCORDBIO",
    "JWL",
    "HONASA",
    "SIGNATURE",
    "FIRSTCRY",
    "PCBL",
    "BLUEDART",
    "HEG",
    "HOMEFIRST",
    "NUVOCO",
    "AFCONS",
    "RPOWER",
    "IEX",
    "CANFINHOME",
    "ZENSARTECH",
    "BLS",
    "ELECON",
    "WHIRLPOOL",
    "AAVAS",
    "BALRAMCHIN",
    "ACE",
    "MGL",
    "BBTC",
    "JKTYRE",
    "RAILTEL",
    "RKFORGE",
    "TITAGARH",
    "CYIENT",
    "SWANCORP",
    "SBFC",
    "OLECTRA",
    "RITES",
    "PVRINOX",
    "INTELLECT",
    "MMTC",
    "NCC",
    "TARIL",
    "BSOFT",
    "BATAINDIA",
    "LEMONTREE",
    "CARTRADE",
    "TEJASNET",
    "TTML",
    "CLEAN",
    "RHIM",
    "ZEEL",
    "SAREGAMA",
    "ABFRL",
    "BLUEJET",
    "SONATSOFTW",
    "LATENTVIEW",
    "NEWGEN",
    "SAPPHIRE",
    "MAPMYINDIA",
]

RAW_NIFTY_50 = [
    "RELIANCE", "HDFCBANK", "TCS", "ICICIBANK", "BHARTIARTL", "SBIN", "INFY",
    "LICI", "HINDUNILVR", "ITC", "LT", "KOTAKBANK", "BAJFINANCE", "AXISBANK",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NTPC", "ONGC", "M&M", "WIPRO",
    "TATASTEEL", "POWERGRID", "ASIANPAINT", "BAJAJFINSV", "ADANIPORTS", "JSWSTEEL",
    "HCLTECH", "NESTLEIND", "TATAMOTORS", "COALINDIA", "TECHM", "GRASIM", "CIPLA",
    "DRREDDY", "ADANIENT", "BRITANNIA", "DIVISLAB", "HEROMOTOCO", "EICHERMOT",
    "APOLLOHOSP", "LTIM", "BAJAJ-AUTO", "INDUSINDBK", "HINDALCO", "SHRIRAMFIN",
    "SBILIFE", "ADANIGREEN", "TRENT",
]

RAW_FN_O = RAW_STOCKS[:120]
SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "TSLA", "LLY",
    "AVGO", "JPM", "V", "XOM", "MA", "NFLX", "COST", "WMT", "UNH", "ORCL",
]
NASDAQ_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "NFLX", "ADBE", "AMD",
    "INTC", "CSCO", "PEP", "COST", "QCOM", "TXN", "INTU", "AMAT", "BKNG", "ISRG",
]


def normalize_nse_ticker(symbol: Any) -> str:
    symbol = str(symbol).strip().upper()
    if not symbol:
        return ""
    if not symbol.endswith(".NS"):
        symbol = symbol + ".NS"
    return symbol


def normalize_generic_ticker(symbol: Any) -> str:
    return str(symbol).strip().upper()


def build_stock_universe(raw_symbols: list[str]) -> list[str]:
    normalized = [normalize_nse_ticker(symbol) for symbol in raw_symbols if str(symbol).strip()]
    return sorted(list(set(normalized)))


STOCKS = build_stock_universe(RAW_STOCKS)
NIFTY_50 = build_stock_universe(RAW_NIFTY_50)
FNO_STOCKS = build_stock_universe(RAW_FN_O)


def safe_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)) and not pd.isna(value):
        return float(value)
    return None


def clamp(value: float, lower: float = 0, upper: float = 100) -> float:
    return float(max(lower, min(upper, value)))


def format_float(value: Any) -> str:
    numeric = safe_number(value)
    return "Not available" if numeric is None else f"{numeric:.2f}"


def format_ratio(value: Any) -> str:
    numeric = safe_number(value)
    return "Not available" if numeric is None else f"{numeric:.2%}"


def format_large_number(value: Any) -> str:
    numeric = safe_number(value)
    return "Not available" if numeric is None else f"{numeric:,.2f}"


def add_scored_reason(bucket: list[dict[str, Any]], factor: str, score: float, reason: str) -> None:
    bucket.append({"Factor": factor, "Score": round(score, 2), "Reason": reason})


def init_session_state() -> None:
    defaults = {
        "logged_in": False,
        "watchlists": {},
        "portfolio_analysis": None,
        "institutional_data": None,
        "institutional_score": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_watchlists() -> dict[str, list[str]]:
    if not WATCHLIST_FILE.exists():
        return {}
    try:
        with WATCHLIST_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return {str(key): [normalize_generic_ticker(item) for item in value] for key, value in data.items()}
    except Exception:
        st.warning("Saved watchlists could not be read, so a blank watchlist store was loaded.")
    return {}


def save_watchlists(watchlists: dict[str, list[str]]) -> None:
    try:
        with WATCHLIST_FILE.open("w", encoding="utf-8") as handle:
            json.dump(watchlists, handle, indent=2)
    except Exception as error:
        st.warning(f"Watchlists could not be saved locally: {error}")


def parse_custom_watchlist(text_value: str) -> list[str]:
    if not text_value.strip():
        return []
    return sorted(list({normalize_generic_ticker(item) for item in text_value.split(",") if item.strip()}))


def refresh_cache_button() -> None:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.success("Cached market data was cleared.")


def normalize_ticker_for_universe(symbol: str, universe: str) -> str:
    if universe in {"Nifty 50", "F&O Stocks"}:
        return normalize_nse_ticker(symbol)
    return normalize_generic_ticker(symbol)


def get_universe_tickers(universe: str, custom_text: str, saved_watchlist: str | None) -> tuple[list[str], str]:
    if universe == "Nifty 50":
        return NIFTY_50, "Using a predefined Nifty 50 universe."
    if universe == "F&O Stocks":
        return FNO_STOCKS, "Using a predefined F&O approximation universe."
    if universe == "S&P 500":
        return SP500_TICKERS, "Using a compact predefined S&P 500 large-cap list for free screening."
    if universe == "NASDAQ":
        return NASDAQ_TICKERS, "Using a compact predefined NASDAQ large-cap list for free screening."
    if universe == "Custom Watchlist":
        tickers = parse_custom_watchlist(custom_text)
        if saved_watchlist and saved_watchlist in st.session_state["watchlists"]:
            tickers = sorted(list(set(tickers + st.session_state["watchlists"][saved_watchlist])))
        return tickers, "Using your custom watchlist input."
    return [], "No universe selected."


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker_symbol: str, period: str) -> pd.DataFrame:
    history = yf.Ticker(ticker_symbol).history(period=period, auto_adjust=False)
    if history.empty:
        raise ValueError(f"No price data was returned for {ticker_symbol}.")
    history = history.reset_index()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
    history = history.dropna(subset=["Open", "High", "Low", "Close"])
    if history.empty:
        raise ValueError(f"{ticker_symbol} returned empty OHLC data after cleanup.")
    return history


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker_symbol: str) -> dict[str, Any]:
    info = yf.Ticker(ticker_symbol).info or {}
    return {
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "priceToBook": info.get("priceToBook"),
        "dividendYield": info.get("dividendYield"),
        "returnOnEquity": info.get("returnOnEquity"),
        "debtToEquity": info.get("debtToEquity"),
        "revenueGrowth": info.get("revenueGrowth"),
        "profitMargins": info.get("profitMargins"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "beta": info.get("beta"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_options_snapshot(ticker_symbol: str) -> tuple[list[str], str | None]:
    ticker = yf.Ticker(ticker_symbol)
    expiries = ticker.options
    if not expiries:
        return [], "Options data is unavailable for this symbol in Yahoo Finance."
    return list(expiries), None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_option_chain(ticker_symbol: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    chain = yf.Ticker(ticker_symbol).option_chain(expiry)
    return chain.calls.copy(), chain.puts.copy()


def calculate_pivot_levels(dataframe: pd.DataFrame) -> pd.DataFrame:
    latest_row = dataframe.iloc[-1]
    high = float(latest_row["High"])
    low = float(latest_row["Low"])
    close = float(latest_row["Close"])
    pivot_point = (high + low + close) / 3
    resistance_1 = (2 * pivot_point) - low
    support_1 = (2 * pivot_point) - high
    resistance_2 = pivot_point + (high - low)
    support_2 = pivot_point - (high - low)
    resistance_3 = high + 2 * (pivot_point - low)
    support_3 = low - 2 * (high - pivot_point)
    return pd.DataFrame(
        {
            "Level": ["Pivot Point", "R1", "R2", "R3", "S1", "S2", "S3"],
            "Value": [pivot_point, resistance_1, resistance_2, resistance_3, support_1, support_2, support_3],
        }
    )


def calculate_rsi(dataframe: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    result = dataframe.copy()
    delta = result["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.rolling(window=period, min_periods=period).mean()
    average_loss = losses.rolling(window=period, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result["RSI"] = 100 - (100 / (1 + relative_strength))
    result.loc[(average_loss == 0) & (average_gain > 0), "RSI"] = 100
    result.loc[(average_gain == 0) & (average_loss > 0), "RSI"] = 0
    result["RSI"] = result["RSI"].fillna(50)
    return result


def calculate_macd(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    fast_ema = result["Close"].ewm(span=12, adjust=False).mean()
    slow_ema = result["Close"].ewm(span=26, adjust=False).mean()
    result["MACD"] = fast_ema - slow_ema
    result["MACD_Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["MACD_Histogram"] = result["MACD"] - result["MACD_Signal"]
    return result


def calculate_mean_reversion(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result["Rolling_Mean_20"] = result["Close"].rolling(window=20, min_periods=20).mean()
    result["Rolling_Std_20"] = result["Close"].rolling(window=20, min_periods=20).std()
    result["Z_Score"] = (
        (result["Close"] - result["Rolling_Mean_20"])
        / result["Rolling_Std_20"].replace(0, np.nan)
    )
    result["Z_Score"] = result["Z_Score"].replace([np.inf, -np.inf], np.nan).fillna(0)
    result["SMA_20"] = result["Close"].rolling(window=20, min_periods=1).mean()
    result["SMA_50"] = result["Close"].rolling(window=50, min_periods=1).mean()
    return result


def compute_return_percent(series: pd.Series, lookback: int) -> float | None:
    if len(series) <= lookback:
        return None
    base = float(series.iloc[-lookback - 1])
    latest = float(series.iloc[-1])
    if base == 0:
        return None
    return ((latest / base) - 1) * 100


def interpret_rsi(rsi_value: float) -> str:
    if rsi_value > 70:
        return "Overbought"
    if rsi_value < 30:
        return "Oversold"
    return "Neutral"


def interpret_macd(macd_value: float, signal_value: float) -> str:
    difference = macd_value - signal_value
    if abs(difference) <= max(0.05, abs(macd_value) * 0.05):
        return "Neutral"
    return "Bullish" if difference > 0 else "Bearish"


def interpret_mean_reversion(z_score: float) -> str:
    if z_score > 2:
        return "Overbought / possible mean reversal downside"
    if z_score < -2:
        return "Oversold / possible mean reversal upside"
    return "Neutral"


def detect_candlestick_patterns(dataframe: pd.DataFrame) -> dict[str, str]:
    if len(dataframe) < 3:
        return {"pattern": "Not enough data", "interpretation": "At least three candles are needed."}

    latest = dataframe.iloc[-1]
    previous = dataframe.iloc[-2]
    third_last = dataframe.iloc[-3]

    body = abs(float(latest["Close"]) - float(latest["Open"]))
    candle_range = max(float(latest["High"]) - float(latest["Low"]), 1e-9)
    upper_shadow = float(latest["High"]) - max(float(latest["Open"]), float(latest["Close"]))
    lower_shadow = min(float(latest["Open"]), float(latest["Close"])) - float(latest["Low"])

    if body <= candle_range * 0.1:
        return {"pattern": "Doji", "interpretation": "Indecision in the latest candle."}
    if lower_shadow > body * 2 and upper_shadow < body:
        return {"pattern": "Hammer", "interpretation": "Possible bullish reversal near support."}
    if upper_shadow > body * 2 and lower_shadow < body:
        return {"pattern": "Shooting Star", "interpretation": "Possible bearish rejection after an upswing."}
    if upper_shadow > body * 2 and float(latest["Close"]) > float(latest["Open"]):
        return {"pattern": "Inverted Hammer", "interpretation": "Potential bullish reversal if confirmed."}
    if body >= candle_range * 0.8:
        return {"pattern": "Marubozu", "interpretation": "Strong directional conviction in the latest candle."}

    previous_bearish = float(previous["Close"]) < float(previous["Open"])
    previous_bullish = float(previous["Close"]) > float(previous["Open"])
    latest_bullish = float(latest["Close"]) > float(latest["Open"])
    latest_bearish = float(latest["Close"]) < float(latest["Open"])

    if previous_bearish and latest_bullish and float(latest["Close"]) > float(previous["Open"]) and float(latest["Open"]) < float(previous["Close"]):
        return {"pattern": "Bullish Engulfing", "interpretation": "Potential bullish reversal after weakness."}
    if previous_bullish and latest_bearish and float(latest["Open"]) > float(previous["Close"]) and float(latest["Close"]) < float(previous["Open"]):
        return {"pattern": "Bearish Engulfing", "interpretation": "Potential bearish reversal after strength."}

    if float(third_last["Close"]) < float(third_last["Open"]) and abs(float(previous["Close"]) - float(previous["Open"])) <= (float(previous["High"]) - float(previous["Low"])) * 0.2 and latest_bullish:
        return {"pattern": "Morning Star", "interpretation": "Possible bullish reversal if follow-through continues."}
    if float(third_last["Close"]) > float(third_last["Open"]) and abs(float(previous["Close"]) - float(previous["Open"])) <= (float(previous["High"]) - float(previous["Low"])) * 0.2 and latest_bearish:
        return {"pattern": "Evening Star", "interpretation": "Possible bearish reversal if weakness continues."}

    return {"pattern": "No strong pattern", "interpretation": "No major candlestick signal was detected in the latest candles."}


def calculate_relative_strength_metrics(close_series: pd.Series) -> dict[str, float | None]:
    returns = {
        "20d_return": compute_return_percent(close_series, 20),
        "50d_return": compute_return_percent(close_series, 50),
        "100d_return": compute_return_percent(close_series, 100),
        "200d_return": compute_return_percent(close_series, 200),
    }
    available_weights = []
    weighted_values = []
    weights = {"20d_return": 0.40, "50d_return": 0.30, "100d_return": 0.20, "200d_return": 0.10}
    for key, weight in weights.items():
        value = returns[key]
        if value is not None:
            available_weights.append(weight)
            weighted_values.append(weight * value)
    if not available_weights:
        score = None
    else:
        score = sum(weighted_values) / sum(available_weights)
    returns["relative_strength_score"] = score
    return returns


def build_options_metrics(calls: pd.DataFrame, puts: pd.DataFrame, spot_price: float | None = None) -> dict[str, Any]:
    calls = calls.fillna(0)
    puts = puts.fillna(0)
    total_put_oi = float(puts["openInterest"].sum())
    total_call_oi = float(calls["openInterest"].sum())
    total_put_volume = float(puts["volume"].sum())
    total_call_volume = float(calls["volume"].sum())
    pcr_oi = total_put_oi / total_call_oi if total_call_oi else None
    pcr_volume = total_put_volume / total_call_volume if total_call_volume else None

    strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
    pain_rows = []
    for strike in strikes:
        call_loss = ((strike - calls["strike"]).clip(lower=0) * calls["openInterest"]).sum()
        put_loss = ((puts["strike"] - strike).clip(lower=0) * puts["openInterest"]).sum()
        pain_rows.append({"strike": strike, "total_pain": float(call_loss + put_loss)})
    max_pain_df = pd.DataFrame(pain_rows)
    max_pain_strike = float(max_pain_df.loc[max_pain_df["total_pain"].idxmin(), "strike"]) if not max_pain_df.empty else None

    if spot_price is None and not calls.empty:
        spot_price = float(calls["strike"].median())
    otm_calls = calls[calls["strike"] > spot_price] if spot_price is not None else calls
    otm_puts = puts[puts["strike"] < spot_price] if spot_price is not None else puts
    avg_otm_call_iv = safe_number(otm_calls["impliedVolatility"].replace(0, np.nan).mean())
    avg_otm_put_iv = safe_number(otm_puts["impliedVolatility"].replace(0, np.nan).mean())
    iv_skew = None
    if avg_otm_call_iv is not None and avg_otm_put_iv is not None:
        iv_skew = avg_otm_put_iv - avg_otm_call_iv

    return {
        "pcr_oi": pcr_oi,
        "pcr_volume": pcr_volume,
        "max_pain": max_pain_strike,
        "iv_skew": iv_skew,
        "avg_otm_call_iv": avg_otm_call_iv,
        "avg_otm_put_iv": avg_otm_put_iv,
        "top_call_oi": calls.nlargest(5, "openInterest")[["strike", "openInterest", "impliedVolatility"]],
        "top_put_oi": puts.nlargest(5, "openInterest")[["strike", "openInterest", "impliedVolatility"]],
        "pain_table": max_pain_df,
    }


def options_sentiment_component(options_metrics: dict[str, Any] | None) -> tuple[float | None, list[str], list[str], list[str]]:
    if not options_metrics:
        return None, [], [], ["Options data unavailable."]
    score = 50.0
    bullish_reasons: list[str] = []
    bearish_risks: list[str] = []
    missing_notes: list[str] = []

    pcr_oi = options_metrics.get("pcr_oi")
    if pcr_oi is not None:
        if pcr_oi < 0.7:
            score += 12
            bullish_reasons.append("PCR by open interest is call-heavy.")
        elif pcr_oi > 1.0:
            score -= 8
            bearish_risks.append("PCR by open interest suggests hedging pressure.")
    else:
        missing_notes.append("PCR by open interest unavailable.")

    iv_skew = options_metrics.get("iv_skew")
    if iv_skew is not None:
        if iv_skew > 0.03:
            score -= 10
            bearish_risks.append("Put IV skew is elevated, showing downside protection demand.")
        elif iv_skew < -0.01:
            score += 6
            bullish_reasons.append("Call IV is richer than put IV, hinting at upside speculation.")
    else:
        missing_notes.append("IV skew unavailable.")

    return clamp(score), bullish_reasons, bearish_risks, missing_notes


def institutional_sentiment_component(institutional_df: pd.DataFrame | None) -> tuple[float | None, list[str], list[str], list[str]]:
    if institutional_df is None or institutional_df.empty:
        return None, [], [], ["No FII/DII activity was provided."]
    latest = institutional_df.sort_values("Date").iloc[-1]
    fii = safe_number(latest["FII_Net"]) or 0
    dii = safe_number(latest["DII_Net"]) or 0
    score = 50.0
    bullish_reasons: list[str] = []
    bearish_risks: list[str] = []
    if fii > 0 and dii > 0:
        score += 18
        bullish_reasons.append("FII and DII are both net buyers.")
    elif fii < 0 and dii > 0:
        score += 4
        bearish_risks.append("Domestic buying offsets foreign selling only partially.")
    elif fii < 0 and dii < 0:
        score -= 18
        bearish_risks.append("FII and DII are both net sellers.")
    elif fii > 0 and dii < 0:
        score += 2
        bullish_reasons.append("Foreign flows are supportive despite domestic caution.")
    return clamp(score), bullish_reasons, bearish_risks, []


def calculate_ai_smartness_score(
    dataframe: pd.DataFrame,
    fundamentals: dict[str, Any],
    pivot_levels: pd.DataFrame,
    options_metrics: dict[str, Any] | None = None,
    institutional_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    latest_row = dataframe.iloc[-1]
    latest_close = float(latest_row["Close"])
    latest_rsi = float(latest_row["RSI"])
    latest_macd = float(latest_row["MACD"])
    latest_signal = float(latest_row["MACD_Signal"])
    latest_z_score = float(latest_row["Z_Score"])
    latest_sma50 = float(latest_row["SMA_50"])

    technical_details: list[dict[str, Any]] = []
    fundamental_details: list[dict[str, Any]] = []
    risk_details: list[dict[str, Any]] = []
    bullish_reasons: list[str] = []
    bearish_risks: list[str] = []
    missing_notes: list[str] = []

    rsi_condition = interpret_rsi(latest_rsi)
    macd_condition = interpret_macd(latest_macd, latest_signal)
    mean_reversion_condition = interpret_mean_reversion(latest_z_score)

    technical_score = 50.0
    if rsi_condition == "Oversold":
        technical_score += 14
        bullish_reasons.append("RSI is in oversold territory.")
    elif rsi_condition == "Overbought":
        technical_score -= 12
        bearish_risks.append("RSI is elevated and may limit upside.")
    add_scored_reason(technical_details, "RSI", clamp(technical_score), f"RSI is {latest_rsi:.2f} ({rsi_condition}).")

    if macd_condition == "Bullish":
        technical_score += 12
        bullish_reasons.append("MACD momentum is bullish.")
    elif macd_condition == "Bearish":
        technical_score -= 12
        bearish_risks.append("MACD momentum is bearish.")
    add_scored_reason(technical_details, "MACD", clamp(technical_score), f"MACD is {macd_condition.lower()}.")

    if mean_reversion_condition.startswith("Oversold"):
        technical_score += 10
        bullish_reasons.append("Price is stretched below its 20-day mean.")
    elif mean_reversion_condition.startswith("Overbought"):
        technical_score -= 10
        bearish_risks.append("Price is stretched above its 20-day mean.")
    add_scored_reason(technical_details, "Mean Reversion", clamp(technical_score), f"Z-score is {latest_z_score:.2f}.")

    trend_ratio = latest_close / latest_sma50 if latest_sma50 else 1
    if trend_ratio > 1.03:
        technical_score += 8
        bullish_reasons.append("Price is above the 50-day trend.")
    elif trend_ratio < 0.97:
        technical_score -= 8
        bearish_risks.append("Price is below the 50-day trend.")
    add_scored_reason(technical_details, "Trend", clamp(technical_score), f"Price vs SMA50 ratio is {trend_ratio:.2f}.")

    pivot_map = dict(zip(pivot_levels["Level"], pivot_levels["Value"]))
    support_1 = float(pivot_map["S1"])
    resistance_1 = float(pivot_map["R1"])
    distance_to_support = ((latest_close - support_1) / latest_close) * 100
    distance_to_resistance = ((resistance_1 - latest_close) / latest_close) * 100
    if 0 <= distance_to_support <= 2:
        technical_score += 6
        bullish_reasons.append("Price is trading close to support.")
    if 0 <= distance_to_resistance <= 2:
        technical_score -= 6
        bearish_risks.append("Price is trading close to resistance.")
    add_scored_reason(
        technical_details,
        "Support/Resistance",
        clamp(technical_score),
        f"Distance to support is {distance_to_support:.2f}% and distance to resistance is {distance_to_resistance:.2f}%.",
    )

    fundamental_score = 50.0
    trailing_pe = safe_number(fundamentals.get("trailingPE"))
    if trailing_pe is None:
        missing_notes.append("Trailing P/E unavailable.")
    elif 0 < trailing_pe <= 20:
        fundamental_score += 12
        bullish_reasons.append("Valuation looks reasonable on trailing earnings.")
    elif trailing_pe > 35:
        fundamental_score -= 10
        bearish_risks.append("Valuation looks expensive on trailing earnings.")
    add_scored_reason(fundamental_details, "Trailing P/E", clamp(fundamental_score), f"Trailing P/E is {format_float(trailing_pe)}.")

    price_to_book = safe_number(fundamentals.get("priceToBook"))
    if price_to_book is None:
        missing_notes.append("Price-to-book unavailable.")
    elif price_to_book <= 3:
        fundamental_score += 8
    elif price_to_book > 6:
        fundamental_score -= 7
    add_scored_reason(fundamental_details, "Price/Book", clamp(fundamental_score), f"Price-to-book is {format_float(price_to_book)}.")

    debt_to_equity = safe_number(fundamentals.get("debtToEquity"))
    if debt_to_equity is None:
        missing_notes.append("Debt-to-equity unavailable.")
    elif debt_to_equity <= 50:
        fundamental_score += 8
    elif debt_to_equity > 120:
        fundamental_score -= 9
        bearish_risks.append("Balance-sheet leverage is relatively high.")
    add_scored_reason(fundamental_details, "Debt/Equity", clamp(fundamental_score), f"Debt-to-equity is {format_float(debt_to_equity)}.")

    profit_margin = safe_number(fundamentals.get("profitMargins"))
    if profit_margin is None:
        missing_notes.append("Profit margin unavailable.")
    elif profit_margin >= 0.20:
        fundamental_score += 10
        bullish_reasons.append("Profitability is strong.")
    elif profit_margin < 0.08:
        fundamental_score -= 8
        bearish_risks.append("Profit margins are thin.")
    add_scored_reason(fundamental_details, "Profit Margin", clamp(fundamental_score), f"Profit margin is {format_ratio(profit_margin)}.")

    revenue_growth = safe_number(fundamentals.get("revenueGrowth"))
    if revenue_growth is None:
        missing_notes.append("Revenue growth unavailable.")
    elif revenue_growth >= 0.15:
        fundamental_score += 10
        bullish_reasons.append("Revenue growth is healthy.")
    elif revenue_growth < 0:
        fundamental_score -= 8
        bearish_risks.append("Revenue growth is negative.")
    add_scored_reason(fundamental_details, "Revenue Growth", clamp(fundamental_score), f"Revenue growth is {format_ratio(revenue_growth)}.")

    return_on_equity = safe_number(fundamentals.get("returnOnEquity"))
    if return_on_equity is None:
        missing_notes.append("Return on equity unavailable.")
    elif return_on_equity >= 0.15:
        fundamental_score += 8
    elif return_on_equity < 0.08:
        fundamental_score -= 7
    add_scored_reason(fundamental_details, "ROE", clamp(fundamental_score), f"ROE is {format_ratio(return_on_equity)}.")

    rs_metrics = calculate_relative_strength_metrics(dataframe["Close"])
    rs_value = rs_metrics["relative_strength_score"]
    if rs_value is None:
        relative_strength_score = None
        missing_notes.append("Relative strength data is incomplete for this ticker.")
    else:
        relative_strength_score = clamp(50 + (rs_value / 2))
        if rs_value > 10:
            bullish_reasons.append("Relative strength is stronger than average.")
        elif rs_value < -10:
            bearish_risks.append("Relative strength is weak.")

    risk_score = 50.0
    daily_returns = dataframe["Close"].pct_change().dropna()
    annualized_volatility = float(daily_returns.std() * np.sqrt(252)) if not daily_returns.empty else 0
    running_max = dataframe["Close"].cummax()
    drawdown_series = (dataframe["Close"] / running_max) - 1
    max_drawdown = float(drawdown_series.min()) if not drawdown_series.empty else 0
    if annualized_volatility <= 0.20:
        risk_score += 10
    elif annualized_volatility > 0.35:
        risk_score -= 12
        bearish_risks.append("Volatility is elevated.")
    add_scored_reason(risk_details, "Volatility", clamp(risk_score), f"Annualized volatility is {annualized_volatility:.2%}.")
    if max_drawdown >= -0.10:
        risk_score += 8
    elif max_drawdown < -0.25:
        risk_score -= 10
        bearish_risks.append("Historical drawdown has been deep.")
    add_scored_reason(risk_details, "Drawdown", clamp(risk_score), f"Maximum drawdown is {max_drawdown:.2%}.")
    if distance_to_support > 5:
        risk_score -= 4
    if 0 <= distance_to_resistance <= 2:
        risk_score -= 4
    add_scored_reason(risk_details, "Price Risk", clamp(risk_score), "Support and resistance spacing is included in the risk score.")

    options_score, options_bull, options_bear, options_missing = options_sentiment_component(options_metrics)
    institutional_score, inst_bull, inst_bear, inst_missing = institutional_sentiment_component(institutional_df)
    bullish_reasons.extend(options_bull + inst_bull)
    bearish_risks.extend(options_bear + inst_bear)
    missing_notes.extend(options_missing + inst_missing)

    components = {
        "Technical Score": {"score": clamp(technical_score), "weight": 0.30},
        "Fundamental Score": {"score": clamp(fundamental_score), "weight": 0.20},
        "Relative Strength Score": {"score": relative_strength_score, "weight": 0.15},
        "Risk Score": {"score": clamp(risk_score), "weight": 0.15},
        "Options Sentiment Score": {"score": options_score, "weight": 0.10},
        "Institutional Activity Score": {"score": institutional_score, "weight": 0.10},
    }
    available_components = {key: value for key, value in components.items() if value["score"] is not None}
    total_weight = sum(item["weight"] for item in available_components.values())
    final_score = 50
    if total_weight > 0:
        final_score = round(
            sum(item["score"] * item["weight"] for item in available_components.values()) / total_weight
        )

    if final_score >= 70:
        recommendation = "Buy"
    elif final_score >= 40:
        recommendation = "Hold"
    else:
        recommendation = "Sell"

    if not bullish_reasons:
        bullish_reasons.append("No strong bullish catalyst was detected.")
    if not bearish_risks:
        bearish_risks.append("No major bearish risk dominated the current setup.")

    component_rows = pd.DataFrame(
        {
            "Component": list(components.keys()),
            "Score": [components[name]["score"] for name in components],
            "Weight": [components[name]["weight"] for name in components],
        }
    )
    return {
        "score": final_score,
        "recommendation": recommendation,
        "technical_breakdown": pd.DataFrame(technical_details),
        "fundamental_breakdown": pd.DataFrame(fundamental_details),
        "risk_breakdown": pd.DataFrame(risk_details),
        "component_breakdown": component_rows,
        "bullish_reasons": bullish_reasons[:6],
        "bearish_risks": bearish_risks[:6],
        "missing_notes": sorted(list(set(missing_notes)))[:8],
        "rsi_condition": rsi_condition,
        "macd_condition": macd_condition,
        "mean_reversion_condition": mean_reversion_condition,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "distance_to_support_pct": distance_to_support,
        "distance_to_resistance_pct": distance_to_resistance,
        "relative_strength_metrics": rs_metrics,
    }


def display_fundamentals(fundamentals: dict[str, Any]) -> None:
    table = pd.DataFrame(
        {
            "Metric": [
                "Market Cap", "Trailing P/E", "Forward P/E", "Price to Book",
                "Dividend Yield", "Return on Equity", "Debt to Equity",
                "Revenue Growth", "Profit Margins", "Beta", "Sector", "Industry",
            ],
            "Value": [
                format_large_number(fundamentals.get("marketCap")),
                format_float(fundamentals.get("trailingPE")),
                format_float(fundamentals.get("forwardPE")),
                format_float(fundamentals.get("priceToBook")),
                format_ratio(fundamentals.get("dividendYield")),
                format_ratio(fundamentals.get("returnOnEquity")),
                format_float(fundamentals.get("debtToEquity")),
                format_ratio(fundamentals.get("revenueGrowth")),
                format_ratio(fundamentals.get("profitMargins")),
                format_float(fundamentals.get("beta")),
                fundamentals.get("sector") or "Not available",
                fundamentals.get("industry") or "Not available",
            ],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def build_price_chart(dataframe: pd.DataFrame, ticker_symbol: str, forecast_df: pd.DataFrame | None = None) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=dataframe["Date"], y=dataframe["Close"], mode="lines", name="Close", line={"color": "#0b6e4f", "width": 3})
    )
    if "SMA_50" in dataframe:
        figure.add_trace(
            go.Scatter(x=dataframe["Date"], y=dataframe["SMA_50"], mode="lines", name="SMA 50", line={"color": "#f4a259", "width": 2, "dash": "dash"})
        )
    if forecast_df is not None and not forecast_df.empty:
        figure.add_trace(
            go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines+markers", name="Forecast", line={"color": "#c1121f", "width": 2})
        )
    figure.update_layout(title=f"{ticker_symbol} Closing Price", template="plotly_white", hovermode="x unified", margin={"l": 10, "r": 10, "t": 50, "b": 10})
    return figure


def build_rsi_chart(dataframe: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=dataframe["Date"], y=dataframe["RSI"], mode="lines", name="RSI", line={"color": "#005f73", "width": 2}))
    figure.add_hline(y=70, line_dash="dash", line_color="#c1121f")
    figure.add_hline(y=30, line_dash="dash", line_color="#2a9d8f")
    figure.update_layout(title="RSI (14)", template="plotly_white", yaxis_title="RSI", margin={"l": 10, "r": 10, "t": 50, "b": 10})
    return figure


def build_macd_chart(dataframe: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": False}]])
    figure.add_trace(go.Bar(x=dataframe["Date"], y=dataframe["MACD_Histogram"], name="Histogram", marker_color="#94d2bd"))
    figure.add_trace(go.Scatter(x=dataframe["Date"], y=dataframe["MACD"], mode="lines", name="MACD", line={"color": "#0a9396", "width": 2}))
    figure.add_trace(go.Scatter(x=dataframe["Date"], y=dataframe["MACD_Signal"], mode="lines", name="Signal", line={"color": "#ee9b00", "width": 2}))
    figure.update_layout(title="MACD", template="plotly_white", barmode="relative", margin={"l": 10, "r": 10, "t": 50, "b": 10})
    return figure


def forecast_garch_volatility(price_data: pd.DataFrame) -> dict[str, Any]:
    returns = price_data["Close"].pct_change().dropna() * 100
    realized_vol = float(price_data["Close"].pct_change().dropna().std() * np.sqrt(252))
    fallback_forecast = price_data["Close"].pct_change().rolling(20).std().iloc[-1]
    fallback_forecast = float((fallback_forecast or 0) * np.sqrt(252))

    try:
        from arch import arch_model

        model = arch_model(returns, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        fitted_model = model.fit(disp="off")
        forecast_variance = fitted_model.forecast(horizon=5).variance.iloc[-1] / (100 ** 2)
        forecast_df = pd.DataFrame(
            {
                "Day": [f"Day {index}" for index in range(1, 6)],
                "Forecast Volatility": np.sqrt(forecast_variance.values) * np.sqrt(252),
            }
        )
        return {
            "realized_volatility": realized_vol,
            "forecast_volatility": float(forecast_df["Forecast Volatility"].mean()),
            "forecast_table": forecast_df,
            "method": "GARCH(1,1)",
            "warning": None,
        }
    except Exception:
        forecast_df = pd.DataFrame(
            {
                "Day": [f"Day {index}" for index in range(1, 6)],
                "Forecast Volatility": [fallback_forecast] * 5,
            }
        )
        return {
            "realized_volatility": realized_vol,
            "forecast_volatility": fallback_forecast,
            "forecast_table": forecast_df,
            "method": "20-day rolling volatility fallback",
            "warning": "The optional `arch` package was unavailable or the GARCH fit failed, so a rolling-volatility fallback is being used.",
        }


def build_forecast_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)


def forecast_arima(price_data: pd.DataFrame, horizon: int) -> dict[str, Any]:
    try:
        from statsmodels.tsa.arima.model import ARIMA

        series = price_data["Close"].astype(float)
        model = ARIMA(series, order=(3, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        forecast_df = pd.DataFrame(
            {"Date": build_forecast_dates(price_data["Date"].iloc[-1], horizon), "Forecast": forecast.values}
        )
        return {"model": "ARIMA", "forecast_table": forecast_df, "warning": None}
    except Exception as error:
        return {"model": "ARIMA", "forecast_table": pd.DataFrame(), "warning": f"ARIMA forecasting failed: {error}"}


def forecast_lstm(price_data: pd.DataFrame, horizon: int) -> dict[str, Any]:
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential

        close_values = price_data["Close"].astype(float).values.reshape(-1, 1)
        if len(close_values) < 80:
            return {"model": "LSTM", "forecast_table": pd.DataFrame(), "warning": "LSTM needs at least 80 price points for a stable demo fit."}
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_values)
        window = 20
        x_train = []
        y_train = []
        for index in range(window, len(scaled)):
            x_train.append(scaled[index - window:index, 0])
            y_train.append(scaled[index, 0])
        x_train = np.array(x_train).reshape(-1, window, 1)
        y_train = np.array(y_train)

        model = Sequential([LSTM(24, input_shape=(window, 1)), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=6, batch_size=16, verbose=0)

        recent_window = scaled[-window:].reshape(1, window, 1)
        forecasts = []
        current_window = recent_window.copy()
        for _ in range(horizon):
            next_scaled = model.predict(current_window, verbose=0)[0][0]
            forecasts.append(next_scaled)
            current_window = np.roll(current_window, -1, axis=1)
            current_window[0, -1, 0] = next_scaled
        forecast_values = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        forecast_df = pd.DataFrame(
            {"Date": build_forecast_dates(price_data["Date"].iloc[-1], horizon), "Forecast": forecast_values}
        )
        return {
            "model": "Experimental LSTM",
            "forecast_table": forecast_df,
            "warning": "LSTM is experimental and may be slow on Streamlit Community Cloud.",
        }
    except Exception as error:
        return {"model": "Experimental LSTM", "forecast_table": pd.DataFrame(), "warning": f"LSTM forecasting is unavailable: {error}"}


def calculate_portfolio_var(price_history: pd.DataFrame, holdings: pd.DataFrame) -> dict[str, Any]:
    if price_history.empty:
        raise ValueError("Portfolio price history is empty.")
    returns = price_history.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("Not enough return history was available to calculate portfolio risk.")
    weight_source = holdings.set_index("Ticker")["Current_Value"]
    weights = weight_source / weight_source.sum()
    aligned_weights = weights.reindex(returns.columns).fillna(0)
    portfolio_daily_returns = returns.mul(aligned_weights, axis=1).sum(axis=1)
    volatility = float(portfolio_daily_returns.std() * np.sqrt(252))
    var_95 = float(np.percentile(portfolio_daily_returns, 5))
    downside_var = float(portfolio_daily_returns[portfolio_daily_returns < 0].mean()) if (portfolio_daily_returns < 0).any() else 0.0
    upside_estimate = float(np.percentile(portfolio_daily_returns, 95))
    return {
        "daily_returns": portfolio_daily_returns,
        "volatility": volatility,
        "var_95": var_95,
        "downside_var": downside_var,
        "upside_estimate": upside_estimate,
    }


def create_portfolio_suggestions(holdings: pd.DataFrame, portfolio_risk: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    suggestions = []
    portfolio_messages: list[str] = []
    for _, row in holdings.iterrows():
        if row["AI_Score"] >= 70:
            action = "Increase / accumulate"
            reason = "High score with supportive technical and fundamental signals."
        elif row["AI_Score"] >= 40:
            action = "Hold / monitor"
            reason = "Balanced score with mixed signals."
        else:
            action = "Reduce / review"
            reason = "Low score reflects weaker momentum, valuation, or risk profile."
        if row["Weight"] > 0.25:
            reason += " Position size is above 25%, so concentration deserves attention."
        suggestions.append(
            {"Ticker": row["Ticker"], "AI Score": row["AI_Score"], "Recommendation": row["Recommendation"], "Suggested Action": action, "Reason": reason}
        )
    if (holdings["Weight"] > 0.25).any():
        portfolio_messages.append("One stock exceeds 25% of the portfolio, which increases concentration risk.")
    if portfolio_risk["volatility"] > 0.30:
        portfolio_messages.append("Portfolio volatility is high relative to a diversified long-term allocation.")
    if portfolio_risk["var_95"] < -0.03:
        portfolio_messages.append("Historical 95% daily VaR is elevated, so downside protection may be worth reviewing.")
    if not portfolio_messages:
        portfolio_messages.append("Portfolio risk looks moderate based on current holdings and historical returns.")
    return pd.DataFrame(suggestions), portfolio_messages


def sector_analysis_table(holdings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    sector_table = pd.DataFrame()
    industry_table = pd.DataFrame()
    if "Sector" in holdings.columns:
        sector_table = holdings.groupby("Sector", dropna=False).agg(
            Current_Value=("Current_Value", "sum"),
            Avg_Return=("Return_Pct", "mean"),
            Avg_Volatility=("Annualized_Volatility", "mean"),
            Avg_AI_Score=("AI_Score", "mean"),
        ).reset_index()
        total_value = sector_table["Current_Value"].sum() or 1
        sector_table["Allocation"] = sector_table["Current_Value"] / total_value
        if (sector_table["Allocation"] > 0.40).any():
            warnings.append("One sector exceeds 40% of portfolio value.")
    if "Industry" in holdings.columns:
        industry_table = holdings.groupby("Industry", dropna=False).agg(
            Current_Value=("Current_Value", "sum"),
            Holdings=("Ticker", "count"),
        ).reset_index()
        if not industry_table.empty and industry_table["Holdings"].max() >= max(3, len(holdings) // 2):
            warnings.append("A large share of holdings sits in one industry.")
    return sector_table, industry_table, warnings


def estimate_kelly_position(ai_score: float, upside_estimate: float, downside_estimate: float) -> dict[str, float]:
    win_probability = 0.40 + (ai_score / 100) * 0.20
    reward = max(upside_estimate, 0.01)
    risk = max(abs(downside_estimate), 0.01)
    reward_risk_ratio = reward / risk
    kelly_fraction = win_probability - ((1 - win_probability) / reward_risk_ratio)
    final_position_size = min(max(kelly_fraction, 0), 0.25)
    return {
        "win_probability": win_probability,
        "reward_risk_ratio": reward_risk_ratio,
        "kelly_fraction": kelly_fraction,
        "half_kelly": final_position_size / 2,
        "max_allocation": final_position_size,
    }


def calculate_portfolio_statistics(returns: pd.DataFrame, weights: np.ndarray) -> tuple[float, float, float]:
    mean_daily_returns = returns.mean().values
    cov_matrix = returns.cov().values
    expected_return = float(np.dot(weights, mean_daily_returns) * 252)
    volatility = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights))))
    sharpe = expected_return / volatility if volatility else 0
    return expected_return, volatility, sharpe


def optimize_markowitz(returns: pd.DataFrame) -> dict[str, Any]:
    asset_count = returns.shape[1]
    if asset_count == 0:
        raise ValueError("No return series were available for optimization.")

    def objective(weights: np.ndarray) -> float:
        expected_return, volatility, sharpe = calculate_portfolio_statistics(returns, weights)
        return -sharpe

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
    bounds = tuple((0, 1) for _ in range(asset_count))
    initial = np.repeat(1 / asset_count, asset_count)
    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else initial
    expected_return, volatility, sharpe = calculate_portfolio_statistics(returns, weights)
    return {"weights": weights, "expected_return": expected_return, "volatility": volatility, "sharpe": sharpe}


def optimize_inverse_volatility(returns: pd.DataFrame) -> dict[str, Any]:
    vol = returns.std() * np.sqrt(252)
    inverse_vol = 1 / vol.replace(0, np.nan)
    weights = (inverse_vol / inverse_vol.sum()).fillna(0).values
    expected_return, volatility, sharpe = calculate_portfolio_statistics(returns, weights)
    return {"weights": weights, "expected_return": expected_return, "volatility": volatility, "sharpe": sharpe}


def optimize_black_litterman_simplified(
    returns: pd.DataFrame,
    current_weights: np.ndarray,
    views: dict[str, float],
) -> dict[str, Any]:
    annual_returns = returns.mean() * 252
    adjusted_returns = annual_returns.copy()
    for ticker, view in views.items():
        if ticker in adjusted_returns.index:
            adjusted_returns.loc[ticker] += view / 100
    cov_matrix = returns.cov() * 252
    try:
        inv_cov = np.linalg.pinv(cov_matrix.values)
        raw_weights = inv_cov.dot(adjusted_returns.values)
        raw_weights = np.clip(raw_weights, 0, None)
        weights = raw_weights / raw_weights.sum() if raw_weights.sum() else current_weights
    except Exception:
        weights = current_weights
    expected_return, volatility, sharpe = calculate_portfolio_statistics(returns, weights)
    return {"weights": weights, "expected_return": expected_return, "volatility": volatility, "sharpe": sharpe}


def run_monte_carlo_simulation(
    returns: pd.DataFrame,
    weights: np.ndarray,
    simulations: int,
    horizon: int,
    starting_value: float,
) -> dict[str, Any]:
    mean_vector = returns.mean().values
    cov_matrix = returns.cov().values
    simulated_paths = np.zeros((horizon, simulations))
    final_values = []
    for simulation in range(simulations):
        portfolio_value = starting_value
        path = []
        draws = np.random.multivariate_normal(mean_vector, cov_matrix, horizon)
        for draw in draws:
            portfolio_return = float(np.dot(draw, weights))
            portfolio_value *= (1 + portfolio_return)
            path.append(portfolio_value)
        simulated_paths[:, simulation] = path
        final_values.append(portfolio_value)
    final_series = pd.Series(final_values)
    return {
        "paths": simulated_paths,
        "final_values": final_series,
        "expected_final_value": float(final_series.mean()),
        "p5": float(final_series.quantile(0.05)),
        "p50": float(final_series.quantile(0.50)),
        "p95": float(final_series.quantile(0.95)),
        "probability_of_loss": float((final_series < starting_value).mean()),
    }


def fetch_and_prepare_analysis(ticker_symbol: str, period: str) -> dict[str, Any]:
    history = fetch_stock_data(ticker_symbol, period)
    history = calculate_rsi(history)
    history = calculate_macd(history)
    history = calculate_mean_reversion(history)
    fundamentals = fetch_fundamentals(ticker_symbol)
    pivot_levels = calculate_pivot_levels(history)
    score_card = calculate_ai_smartness_score(
        history,
        fundamentals,
        pivot_levels,
        institutional_df=st.session_state.get("institutional_data"),
    )
    pattern = detect_candlestick_patterns(history)
    volatility_forecast = forecast_garch_volatility(history)
    return {
        "history": history,
        "fundamentals": fundamentals,
        "pivot_levels": pivot_levels,
        "score_card": score_card,
        "pattern": pattern,
        "volatility_forecast": volatility_forecast,
    }


def analyze_portfolio(portfolio_file: Any, period: str) -> dict[str, Any] | None:
    if portfolio_file is None:
        return None

    portfolio_df = pd.read_csv(portfolio_file)
    missing_columns = [column for column in REQUIRED_PORTFOLIO_COLUMNS if column not in portfolio_df.columns]
    if missing_columns:
        raise ValueError("Portfolio CSV is missing required columns: " + ", ".join(missing_columns))

    working_df = portfolio_df[REQUIRED_PORTFOLIO_COLUMNS].copy()
    working_df["Ticker"] = working_df["Ticker"].apply(normalize_nse_ticker)
    working_df["Quantity"] = pd.to_numeric(working_df["Quantity"], errors="coerce")
    working_df["Average_Buy_Price"] = pd.to_numeric(working_df["Average_Buy_Price"], errors="coerce")

    if working_df["Ticker"].eq("").any() or working_df["Ticker"].eq(".NS").any():
        raise ValueError("Portfolio CSV contains an empty ticker value.")
    if working_df["Quantity"].isna().any() or working_df["Average_Buy_Price"].isna().any():
        raise ValueError("Quantity and Average_Buy_Price must be numeric for every row.")
    if (working_df["Quantity"] <= 0).any() or (working_df["Average_Buy_Price"] <= 0).any():
        raise ValueError("Quantity and Average_Buy_Price must be greater than zero for every holding.")

    holdings_rows = []
    price_series_map: dict[str, pd.Series] = {}
    issues: list[str] = []

    for _, row in working_df.iterrows():
        ticker = row["Ticker"]
        try:
            analysis = fetch_and_prepare_analysis(ticker, period)
            history = analysis["history"]
            fundamentals = analysis["fundamentals"]
            score_card = analysis["score_card"]
            latest_close = float(history["Close"].iloc[-1])
            invested_value = float(row["Quantity"] * row["Average_Buy_Price"])
            current_value = float(row["Quantity"] * latest_close)
            profit_loss = current_value - invested_value
            return_pct = (profit_loss / invested_value) * 100 if invested_value else 0
            annualized_volatility = float(history["Close"].pct_change().dropna().std() * np.sqrt(252))
            rs_metrics = score_card["relative_strength_metrics"]

            holdings_rows.append(
                {
                    "Ticker": ticker,
                    "Quantity": float(row["Quantity"]),
                    "Average_Buy_Price": float(row["Average_Buy_Price"]),
                    "Latest_Price": latest_close,
                    "Invested_Value": invested_value,
                    "Current_Value": current_value,
                    "Unrealized_PnL": profit_loss,
                    "Return_Pct": return_pct,
                    "AI_Score": score_card["score"],
                    "Recommendation": score_card["recommendation"],
                    "Sector": fundamentals.get("sector") or "Unknown",
                    "Industry": fundamentals.get("industry") or "Unknown",
                    "Annualized_Volatility": annualized_volatility,
                    "Relative_Strength": rs_metrics.get("relative_strength_score"),
                }
            )
            price_series_map[ticker] = history.set_index("Date")["Close"].rename(ticker)
        except Exception as error:
            issues.append(f"{ticker}: {error}")

    if not holdings_rows:
        raise ValueError("No portfolio rows could be analyzed. Please verify the tickers and try again.")

    holdings = pd.DataFrame(holdings_rows)
    holdings["Weight"] = holdings["Current_Value"] / holdings["Current_Value"].sum()
    summary = {
        "Total Invested Value": float(holdings["Invested_Value"].sum()),
        "Current Portfolio Value": float(holdings["Current_Value"].sum()),
        "Total Profit / Loss": float(holdings["Unrealized_PnL"].sum()),
    }
    summary["Total Return Percentage"] = (
        (summary["Total Profit / Loss"] / summary["Total Invested Value"]) * 100 if summary["Total Invested Value"] else 0
    )
    price_history = pd.concat(price_series_map.values(), axis=1).sort_index().ffill().dropna(how="all")
    portfolio_risk = calculate_portfolio_var(price_history, holdings)
    suggestions_df, portfolio_messages = create_portfolio_suggestions(holdings, portfolio_risk)
    sector_table, industry_table, concentration_warnings = sector_analysis_table(holdings)

    return {
        "holdings": holdings,
        "summary": summary,
        "risk": portfolio_risk,
        "suggestions": suggestions_df,
        "portfolio_messages": portfolio_messages,
        "sector_table": sector_table,
        "industry_table": industry_table,
        "concentration_warnings": concentration_warnings,
        "price_history": price_history,
        "issues": issues,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def run_screener_analysis(
    tickers: tuple[str, ...],
    max_stocks: int,
    institutional_json: str | None = None,
) -> pd.DataFrame:
    results = []
    institutional_df = pd.read_json(institutional_json) if institutional_json else None
    for ticker in list(tickers)[:max_stocks]:
        try:
            history = fetch_stock_data(ticker, "1y")
            history = calculate_rsi(history)
            history = calculate_macd(history)
            history = calculate_mean_reversion(history)
            fundamentals = fetch_fundamentals(ticker)
            pivot_levels = calculate_pivot_levels(history)
            score_card = calculate_ai_smartness_score(history, fundamentals, pivot_levels, institutional_df=institutional_df)
            close_series = history["Close"]
            latest_close = float(close_series.iloc[-1])
            daily_return = float(close_series.pct_change().iloc[-1] * 100) if len(close_series) > 1 else 0
            return_20 = compute_return_percent(close_series, 20)
            return_50 = compute_return_percent(close_series, 50)
            rs_metrics = score_card["relative_strength_metrics"]
            week_52_high = fundamentals.get("fiftyTwoWeekHigh") or history["High"].tail(min(len(history), 252)).max()
            week_52_low = fundamentals.get("fiftyTwoWeekLow") or history["Low"].tail(min(len(history), 252)).min()
            dist_high = ((latest_close / week_52_high) - 1) * 100 if week_52_high else None
            dist_low = ((latest_close / week_52_low) - 1) * 100 if week_52_low else None
            pattern = detect_candlestick_patterns(history)
            results.append(
                {
                    "Ticker": ticker,
                    "Latest Close": latest_close,
                    "Daily Return %": daily_return,
                    "20D Return %": return_20,
                    "50D Return %": return_50,
                    "100D Return %": rs_metrics.get("100d_return"),
                    "200D Return %": rs_metrics.get("200d_return"),
                    "20D Volatility %": float(history["Close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100),
                    "RSI": float(history["RSI"].iloc[-1]),
                    "MACD Condition": score_card["macd_condition"],
                    "Z-Score": float(history["Z_Score"].iloc[-1]),
                    "Distance from 52W High %": dist_high,
                    "Distance from 52W Low %": dist_low,
                    "Relative Strength Score": rs_metrics.get("relative_strength_score"),
                    "AI Smartness Score": score_card["score"],
                    "Recommendation": score_card["recommendation"],
                    "Sector": fundamentals.get("sector") or "Unknown",
                    "Industry": fundamentals.get("industry") or "Unknown",
                    "Candlestick Pattern": pattern["pattern"],
                    "Pattern Interpretation": pattern["interpretation"],
                }
            )
        except Exception:
            continue
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by="AI Smartness Score", ascending=False)
    return results_df


def render_authentication_sidebar() -> None:
    st.sidebar.subheader("Demo Login")
    st.sidebar.caption("This is demo authentication and local file-based storage only. It is not suitable for sensitive production use.")
    if not st.session_state["logged_in"]:
        username = st.sidebar.text_input("Username", value="demo")
        password = st.sidebar.text_input("Password", type="password", value="demo123")
        if st.sidebar.button("Login"):
            if username == DEMO_USERNAME and password == DEMO_PASSWORD:
                st.session_state["logged_in"] = True
                st.success("Demo login successful.")
            else:
                st.sidebar.error("Invalid demo credentials.")
    else:
        st.sidebar.success("Logged in as demo")
        if st.sidebar.button("Logout"):
            st.session_state["logged_in"] = False
            st.rerun()


def render_watchlist_manager() -> None:
    st.subheader("Saved Watchlists")
    st.caption("This feature uses local JSON storage for demo purposes only.")
    watchlists = st.session_state["watchlists"]
    if st.session_state["logged_in"]:
        watchlist_name = st.text_input("Watchlist name")
        watchlist_tickers = st.text_area("Comma-separated tickers", placeholder="RELIANCE.NS,TCS.NS,AAPL,MSFT")
        if st.button("Save watchlist"):
            parsed = parse_custom_watchlist(watchlist_tickers)
            if watchlist_name.strip() and parsed:
                watchlists[watchlist_name.strip()] = parsed
                st.session_state["watchlists"] = watchlists
                save_watchlists(watchlists)
                st.success("Watchlist saved locally.")
            else:
                st.warning("Enter a watchlist name and at least one valid ticker.")
    else:
        st.info("Login with the demo account to save watchlists.")
    if watchlists:
        display_rows = [{"Watchlist": name, "Tickers": ", ".join(tickers)} for name, tickers in watchlists.items()]
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No watchlists are saved yet.")


def render_market_activity_tab() -> None:
    st.subheader("FII / DII Market Activity")
    source = st.radio("Choose data source", ["Manual Entry", "CSV Upload"], horizontal=True)
    institutional_df: pd.DataFrame | None = None
    if source == "Manual Entry":
        col1, col2, col3 = st.columns(3)
        activity_date = col1.date_input("Date")
        fii_net = col2.number_input("FII Net Buy/Sell Value", value=0.0)
        dii_net = col3.number_input("DII Net Buy/Sell Value", value=0.0)
        if st.button("Use manual institutional activity"):
            institutional_df = pd.DataFrame({"Date": [pd.to_datetime(activity_date)], "FII_Net": [fii_net], "DII_Net": [dii_net]})
            st.session_state["institutional_data"] = institutional_df
    else:
        fii_dii_file = st.file_uploader("Upload FII/DII CSV", type=["csv"], key="fii_dii_uploader")
        if fii_dii_file is not None:
            try:
                uploaded_df = pd.read_csv(fii_dii_file)
                missing = [column for column in REQUIRED_FII_DII_COLUMNS if column not in uploaded_df.columns]
                if missing:
                    st.warning("FII/DII CSV is missing required columns: " + ", ".join(missing))
                else:
                    uploaded_df["Date"] = pd.to_datetime(uploaded_df["Date"])
                    institutional_df = uploaded_df[REQUIRED_FII_DII_COLUMNS].copy()
                    st.session_state["institutional_data"] = institutional_df
            except Exception as error:
                st.error(f"FII/DII CSV could not be processed: {error}")
    institutional_df = st.session_state.get("institutional_data")
    if institutional_df is not None and not institutional_df.empty:
        chart = go.Figure()
        chart.add_trace(go.Bar(x=institutional_df["Date"], y=institutional_df["FII_Net"], name="FII Net"))
        chart.add_trace(go.Bar(x=institutional_df["Date"], y=institutional_df["DII_Net"], name="DII Net"))
        chart.update_layout(barmode="group", template="plotly_white", title="Institutional Activity")
        st.plotly_chart(chart, use_container_width=True)
        combined = institutional_df["FII_Net"] + institutional_df["DII_Net"]
        latest = institutional_df.sort_values("Date").iloc[-1]
        fii = safe_number(latest["FII_Net"]) or 0
        dii = safe_number(latest["DII_Net"]) or 0
        if fii > 0 and dii > 0:
            interpretation = "FII buying + DII buying = Strong institutional support"
        elif fii < 0 and dii > 0:
            interpretation = "FII selling + DII buying = Domestic support but foreign weakness"
        elif fii < 0 and dii < 0:
            interpretation = "FII selling + DII selling = Broad institutional weakness"
        else:
            interpretation = "FII buying + DII selling = Foreign support but domestic caution"
        st.info(interpretation)
        st.dataframe(pd.DataFrame({"Date": institutional_df["Date"], "Combined Activity": combined}), use_container_width=True, hide_index=True)
    else:
        st.info("No institutional activity was provided. The related score component will be skipped safely.")


def render_single_stock_tab(selected_ticker: str, selected_period: str) -> None:
    analysis = fetch_and_prepare_analysis(selected_ticker, selected_period)
    history = analysis["history"]
    fundamentals = analysis["fundamentals"]
    pivot_levels = analysis["pivot_levels"]
    score_card = analysis["score_card"]
    pattern = analysis["pattern"]
    volatility_forecast = analysis["volatility_forecast"]

    latest_row = history.iloc[-1]
    latest_close = float(latest_row["Close"])
    start_date = history["Date"].min().date()
    end_date = history["Date"].max().date()
    daily_change = float(history["Close"].pct_change().iloc[-1] * 100) if len(history) > 1 else 0
    week_52_high = fundamentals.get("fiftyTwoWeekHigh") or history["High"].tail(min(len(history), 252)).max()
    week_52_low = fundamentals.get("fiftyTwoWeekLow") or history["Low"].tail(min(len(history), 252)).min()

    summary_columns = st.columns(5)
    summary_columns[0].metric("Latest Close", f"{latest_close:,.2f}")
    summary_columns[1].metric("Daily Change", f"{daily_change:,.2f}%")
    summary_columns[2].metric("52 Week High", f"{float(week_52_high):,.2f}" if week_52_high else "N/A")
    summary_columns[3].metric("52 Week Low", f"{float(week_52_low):,.2f}" if week_52_low else "N/A")
    summary_columns[4].metric("AI Score", f"{score_card['score']}/100", score_card["recommendation"])

    st.plotly_chart(build_price_chart(history, selected_ticker), use_container_width=True)
    left_chart, right_chart = st.columns(2)
    left_chart.plotly_chart(build_rsi_chart(history), use_container_width=True)
    right_chart.plotly_chart(build_macd_chart(history), use_container_width=True)

    indicator_columns = st.columns(4)
    indicator_columns[0].metric("RSI", f"{latest_row['RSI']:.2f}", score_card["rsi_condition"])
    indicator_columns[1].metric("MACD", f"{latest_row['MACD']:.2f}", score_card["macd_condition"])
    indicator_columns[2].metric("Z-Score", f"{latest_row['Z_Score']:.2f}", score_card["mean_reversion_condition"])
    indicator_columns[3].metric("Range", f"{start_date} to {end_date}")

    pattern_col, vol_col = st.columns(2)
    pattern_col.info(f"{pattern['pattern']}: {pattern['interpretation']}")
    vol_col.info(f"Volatility method: {volatility_forecast['method']}")
    if volatility_forecast["warning"]:
        st.warning(volatility_forecast["warning"])

    volatility_columns = st.columns(2)
    volatility_columns[0].metric("Realized Volatility", f"{volatility_forecast['realized_volatility']:.2%}")
    volatility_columns[1].metric("Forecast Volatility", f"{volatility_forecast['forecast_volatility']:.2%}")

    pivot_and_component = st.columns(2)
    formatted_pivots = pivot_levels.copy()
    formatted_pivots["Value"] = formatted_pivots["Value"].map(lambda value: f"{value:,.2f}")
    pivot_and_component[0].subheader("Pivot Levels")
    pivot_and_component[0].dataframe(formatted_pivots, use_container_width=True, hide_index=True)
    pivot_and_component[1].subheader("AI Component Breakdown")
    pivot_and_component[1].dataframe(score_card["component_breakdown"], use_container_width=True, hide_index=True)

    st.subheader("Fundamental Analysis")
    display_fundamentals(fundamentals)

    st.subheader("Bullish Reasons")
    for reason in score_card["bullish_reasons"]:
        st.write(f"- {reason}")
    st.subheader("Bearish Risks")
    for reason in score_card["bearish_risks"]:
        st.write(f"- {reason}")
    if score_card["missing_notes"]:
        st.subheader("Missing Data Notes")
        for note in score_card["missing_notes"]:
            st.write(f"- {note}")

    detail_tabs = st.tabs(["Technical Breakdown", "Fundamental Breakdown", "Risk Breakdown", "Volatility Forecast"])
    detail_tabs[0].dataframe(score_card["technical_breakdown"], use_container_width=True, hide_index=True)
    detail_tabs[1].dataframe(score_card["fundamental_breakdown"], use_container_width=True, hide_index=True)
    detail_tabs[2].dataframe(score_card["risk_breakdown"], use_container_width=True, hide_index=True)
    detail_tabs[3].dataframe(volatility_forecast["forecast_table"], use_container_width=True, hide_index=True)


def render_portfolio_tab(selected_period: str) -> None:
    st.write("Upload a CSV with `Ticker`, `Quantity`, and `Average_Buy_Price` columns.")
    st.caption("Portfolio tickers can be entered with or without the `.NS` suffix.")
    portfolio_file = st.file_uploader("Upload portfolio CSV", type=["csv"], key="portfolio_uploader")
    if portfolio_file is None:
        st.info("Upload a portfolio file to see live analysis, sector concentration, and optimization inputs.")
        return
    analysis = analyze_portfolio(portfolio_file, selected_period)
    st.session_state["portfolio_analysis"] = analysis
    summary = analysis["summary"]
    holdings = analysis["holdings"]
    risk = analysis["risk"]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Invested", f"{summary['Total Invested Value']:,.2f}")
    metric_columns[1].metric("Current Value", f"{summary['Current Portfolio Value']:,.2f}")
    metric_columns[2].metric("Total P/L", f"{summary['Total Profit / Loss']:,.2f}")
    metric_columns[3].metric("Return %", f"{summary['Total Return Percentage']:,.2f}%")

    risk_columns = st.columns(4)
    risk_columns[0].metric("Portfolio Volatility", f"{risk['volatility']:.2%}")
    risk_columns[1].metric("95% Historical VaR", f"{risk['var_95']:.2%}")
    risk_columns[2].metric("Expected Downside VaR", f"{risk['downside_var']:.2%}")
    risk_columns[3].metric("Upside Estimate", f"{risk['upside_estimate']:.2%}")

    portfolio_view = holdings.copy()
    portfolio_view["Weight"] = portfolio_view["Weight"].map(lambda value: f"{value:.2%}")
    portfolio_view["Return_Pct"] = portfolio_view["Return_Pct"].map(lambda value: f"{value:.2f}%")
    st.subheader("Portfolio Summary Table")
    st.dataframe(portfolio_view, use_container_width=True, hide_index=True)

    st.subheader("Individual Stock Recommendations")
    st.dataframe(analysis["suggestions"], use_container_width=True, hide_index=True)

    st.subheader("Portfolio-Level Recommendation")
    for message in analysis["portfolio_messages"] + analysis["concentration_warnings"]:
        st.write(f"- {message}")

    if not analysis["sector_table"].empty:
        sector_chart = go.Figure(
            data=[go.Pie(labels=analysis["sector_table"]["Sector"], values=analysis["sector_table"]["Allocation"], hole=0.45)]
        )
        sector_chart.update_layout(title="Sector Allocation", template="plotly_white")
        st.plotly_chart(sector_chart, use_container_width=True)
        st.dataframe(analysis["sector_table"], use_container_width=True, hide_index=True)
    if not analysis["industry_table"].empty:
        st.subheader("Industry Allocation")
        st.dataframe(analysis["industry_table"], use_container_width=True, hide_index=True)

    allocation_chart = go.Figure()
    allocation_chart.add_trace(go.Bar(x=holdings["Ticker"], y=holdings["Weight"], name="Weight"))
    allocation_chart.update_layout(title="Portfolio Allocation", template="plotly_white", yaxis_tickformat=".0%")
    st.plotly_chart(allocation_chart, use_container_width=True)

    returns_chart = go.Figure()
    returns_chart.add_trace(go.Scatter(x=risk["daily_returns"].index, y=risk["daily_returns"].cumsum(), mode="lines", name="Cumulative Daily Return", line={"color": "#386641", "width": 3}))
    returns_chart.update_layout(title="Portfolio Cumulative Daily Return", template="plotly_white")
    st.plotly_chart(returns_chart, use_container_width=True)

    if analysis["issues"]:
        st.warning("Some holdings could not be analyzed:")
        for issue in analysis["issues"]:
            st.write(f"- {issue}")


def render_screener_tab() -> None:
    st.subheader("Multi-Stock Screener")
    universe = st.selectbox("Market universe", ["Nifty 50", "F&O Stocks", "S&P 500", "NASDAQ", "Custom Watchlist"])
    custom_text = st.text_area("Custom watchlist tickers", placeholder="RELIANCE.NS,TCS.NS,AAPL,MSFT,NVDA")
    saved_watchlist_name = None
    if st.session_state["watchlists"]:
        saved_watchlist_name = st.selectbox("Use saved watchlist", [""] + list(st.session_state["watchlists"].keys()))
        saved_watchlist_name = saved_watchlist_name or None
    tickers, note = get_universe_tickers(universe, custom_text, saved_watchlist_name)
    st.caption(note)

    max_stocks = st.number_input("Maximum stocks to analyze", min_value=5, max_value=50, value=min(25, max(len(tickers), 25)))
    recommendation_filter = st.selectbox("Recommendation filter", ["All", "Buy", "Hold", "Sell"])
    min_score = st.slider("Minimum AI Smartness Score", 0, 100, 40)
    rsi_range = st.slider("RSI range", 0, 100, (0, 100))
    st.info("Large universes take time on free data sources. The app limits screeners to 50 symbols by default.")

    if len(tickers) > 50:
        st.warning("This universe is larger than the safe default. Only the first 50 symbols will be screened.")

    if st.button("Run Screener"):
        if not tickers:
            st.warning("Please select a universe or enter custom tickers.")
            return
        results_df = run_screener_analysis(tuple(tickers), int(max_stocks), None)
        if results_df.empty:
            st.warning("No screener results were returned. Try a smaller or different universe.")
            return
        filtered = results_df.copy()
        filtered = filtered[filtered["AI Smartness Score"] >= min_score]
        filtered = filtered[filtered["RSI"].between(rsi_range[0], rsi_range[1])]
        if recommendation_filter != "All":
            filtered = filtered[filtered["Recommendation"] == recommendation_filter]
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        ranking_df = filtered.sort_values(by="Relative Strength Score", ascending=False).reset_index(drop=True)
        ranking_df["Rank"] = ranking_df.index + 1
        st.subheader("Relative Strength Ranking")
        st.dataframe(
            ranking_df[["Rank", "Ticker", "20D Return %", "50D Return %", "100D Return %", "200D Return %", "Relative Strength Score", "AI Smartness Score", "Recommendation"]],
            use_container_width=True,
            hide_index=True,
        )
        top_10 = ranking_df.head(10).sort_values(by="Relative Strength Score")
        rs_chart = go.Figure()
        rs_chart.add_trace(go.Bar(x=top_10["Relative Strength Score"], y=top_10["Ticker"], orientation="h", name="RS Score"))
        rs_chart.update_layout(title="Top 10 Relative Strength", template="plotly_white")
        st.plotly_chart(rs_chart, use_container_width=True)

        sector_summary = filtered.groupby("Sector", dropna=False).agg(
            Count=("Ticker", "count"),
            Avg_Return=("20D Return %", "mean"),
            Avg_Volatility=("20D Volatility %", "mean"),
            Avg_AI_Score=("AI Smartness Score", "mean"),
        ).reset_index()
        st.subheader("Sector and Industry Summary")
        st.dataframe(sector_summary, use_container_width=True, hide_index=True)


def render_options_tab() -> None:
    st.subheader("Options Chain Analysis")
    ticker = st.selectbox("Select ticker for options", STOCKS, index=0)
    expiries, warning_message = fetch_options_snapshot(ticker)
    if warning_message:
        st.warning(warning_message)
        return
    expiry = st.selectbox("Select expiry date", expiries)
    try:
        calls, puts = fetch_option_chain(ticker, expiry)
        spot_price = float(fetch_stock_data(ticker, "1mo")["Close"].iloc[-1])
        options_metrics = build_options_metrics(calls, puts, spot_price)
        columns = st.columns(4)
        columns[0].metric("PCR OI", format_float(options_metrics["pcr_oi"]))
        columns[1].metric("PCR Volume", format_float(options_metrics["pcr_volume"]))
        columns[2].metric("Max Pain", format_float(options_metrics["max_pain"]))
        columns[3].metric("IV Skew", format_float(options_metrics["iv_skew"]))
        st.info("PCR above 1 can indicate bearish or hedging-heavy sentiment. PCR below 0.7 can indicate bullish or call-heavy sentiment.")
        st.info("High put IV skew often points to downside protection demand. Max pain is the strike where option sellers may benefit most at expiry.")

        oi_chart = go.Figure()
        oi_chart.add_trace(go.Bar(x=options_metrics["top_call_oi"]["strike"], y=options_metrics["top_call_oi"]["openInterest"], name="Top Call OI"))
        oi_chart.add_trace(go.Bar(x=options_metrics["top_put_oi"]["strike"], y=options_metrics["top_put_oi"]["openInterest"], name="Top Put OI"))
        oi_chart.update_layout(title="Top Open Interest Strikes", template="plotly_white", barmode="group")
        st.plotly_chart(oi_chart, use_container_width=True)

        option_tabs = st.tabs(["Calls", "Puts", "Top OI", "Max Pain Table"])
        option_tabs[0].dataframe(calls, use_container_width=True)
        option_tabs[1].dataframe(puts, use_container_width=True)
        option_tabs[2].write("Highest Call Open Interest Strikes")
        option_tabs[2].dataframe(options_metrics["top_call_oi"], use_container_width=True, hide_index=True)
        option_tabs[2].write("Highest Put Open Interest Strikes")
        option_tabs[2].dataframe(options_metrics["top_put_oi"], use_container_width=True, hide_index=True)
        option_tabs[3].dataframe(options_metrics["pain_table"], use_container_width=True, hide_index=True)
    except Exception as error:
        st.warning(f"Options data could not be processed for {ticker}: {error}")


def render_forecasting_tab() -> None:
    st.subheader("Forecasting")
    ticker = st.selectbox("Select ticker for forecasting", STOCKS, index=0, key="forecast_ticker")
    horizon = st.selectbox("Forecast horizon", FORECAST_HORIZONS, index=0)
    enable_lstm = st.checkbox("Enable experimental LSTM forecast")
    try:
        history = fetch_stock_data(ticker, "2y")
        history = calculate_mean_reversion(history)
    except Exception as error:
        st.error(f"Price history could not be loaded for forecasting: {error}")
        return

    arima_result = forecast_arima(history, horizon)
    forecast_result = arima_result
    if enable_lstm:
        lstm_result = forecast_lstm(history, horizon)
        if not lstm_result["forecast_table"].empty:
            forecast_result = lstm_result
        elif lstm_result["warning"]:
            st.warning(lstm_result["warning"])
    if forecast_result["warning"]:
        st.warning(forecast_result["warning"])
    if forecast_result["forecast_table"].empty:
        st.info("Forecast output is unavailable for this run. Please try another ticker or shorter horizon.")
        return

    st.plotly_chart(build_price_chart(history, ticker, forecast_result["forecast_table"]), use_container_width=True)
    st.metric("Model Used", forecast_result["model"])
    st.dataframe(forecast_result["forecast_table"], use_container_width=True, hide_index=True)
    volatility_forecast = forecast_garch_volatility(history)
    st.subheader("Volatility Forecast")
    if volatility_forecast["warning"]:
        st.warning(volatility_forecast["warning"])
    vol_cols = st.columns(2)
    vol_cols[0].metric("Latest Realized Volatility", f"{volatility_forecast['realized_volatility']:.2%}")
    vol_cols[1].metric("Forecast Volatility", f"{volatility_forecast['forecast_volatility']:.2%}")
    st.dataframe(volatility_forecast["forecast_table"], use_container_width=True, hide_index=True)
    st.info("Forecasts are statistical estimates, not guaranteed predictions.")


def render_portfolio_optimization_tab() -> None:
    st.subheader("Portfolio Optimization")
    portfolio_analysis = st.session_state.get("portfolio_analysis")
    uploaded = st.file_uploader("Upload a portfolio if one is not already loaded", type=["csv"], key="optimizer_portfolio")
    if uploaded is not None:
        try:
            portfolio_analysis = analyze_portfolio(uploaded, "1y")
            st.session_state["portfolio_analysis"] = portfolio_analysis
        except Exception as error:
            st.error(f"Portfolio file could not be analyzed: {error}")
            portfolio_analysis = None
    if not portfolio_analysis:
        st.info("Upload a portfolio in the portfolio tab or here to enable optimization.")
        return

    holdings = portfolio_analysis["holdings"]
    price_history = portfolio_analysis["price_history"]
    returns = price_history.pct_change().dropna()
    current_weights = holdings.set_index("Ticker")["Weight"].reindex(returns.columns).fillna(0).values
    optimizer_method = st.selectbox("Optimization method", ["Markowitz Mean-Variance", "Hierarchical Risk Parity approximation", "Black-Litterman simplified approximation"])

    views: dict[str, float] = {}
    if optimizer_method == "Black-Litterman simplified approximation":
        st.caption("Add optional views such as positive values for expected outperformance and negative values for underperformance.")
        for ticker in holdings["Ticker"]:
            views[ticker] = st.number_input(f"View for {ticker} (%)", value=0.0, key=f"view_{ticker}")

    try:
        if optimizer_method == "Markowitz Mean-Variance":
            optimization = optimize_markowitz(returns)
        elif optimizer_method == "Hierarchical Risk Parity approximation":
            st.info("A full HRP stack is approximated here with inverse-volatility weighting for Streamlit Cloud friendliness.")
            optimization = optimize_inverse_volatility(returns)
        else:
            optimization = optimize_black_litterman_simplified(returns, current_weights, views)
    except Exception as error:
        st.error(f"Optimization failed: {error}")
        return

    optimized_weights = pd.Series(optimization["weights"], index=returns.columns, name="Optimized Weight")
    current_weight_series = pd.Series(current_weights, index=returns.columns, name="Current Weight")
    comparison = pd.concat([current_weight_series, optimized_weights], axis=1).fillna(0)
    comparison["Allocation Difference"] = comparison["Optimized Weight"] - comparison["Current Weight"]
    st.dataframe(comparison.reset_index().rename(columns={"index": "Ticker"}), use_container_width=True, hide_index=True)

    stats_cols = st.columns(3)
    stats_cols[0].metric("Expected Return", f"{optimization['expected_return']:.2%}")
    stats_cols[1].metric("Expected Volatility", f"{optimization['volatility']:.2%}")
    stats_cols[2].metric("Sharpe Ratio", f"{optimization['sharpe']:.2f}")

    comparison_chart = go.Figure()
    comparison_chart.add_trace(go.Bar(x=comparison.index, y=comparison["Current Weight"], name="Current Weight"))
    comparison_chart.add_trace(go.Bar(x=comparison.index, y=comparison["Optimized Weight"], name="Optimized Weight"))
    comparison_chart.update_layout(title="Current vs Optimized Allocation", template="plotly_white", barmode="group")
    st.plotly_chart(comparison_chart, use_container_width=True)

    st.subheader("Monte Carlo Portfolio Simulation")
    simulations = int(st.number_input("Number of simulations", min_value=100, max_value=5000, value=500, step=100))
    horizon = int(st.number_input("Time horizon (trading days)", min_value=20, max_value=252, value=126, step=10))
    starting_value = float(st.number_input("Starting portfolio value", min_value=1000.0, value=float(portfolio_analysis["summary"]["Current Portfolio Value"])))
    if st.button("Run Monte Carlo Simulation"):
        try:
            monte_carlo = run_monte_carlo_simulation(returns, optimization["weights"], simulations, horizon, starting_value)
            mc_cols = st.columns(5)
            mc_cols[0].metric("Expected Final Value", f"{monte_carlo['expected_final_value']:,.2f}")
            mc_cols[1].metric("5th Percentile", f"{monte_carlo['p5']:,.2f}")
            mc_cols[2].metric("50th Percentile", f"{monte_carlo['p50']:,.2f}")
            mc_cols[3].metric("95th Percentile", f"{monte_carlo['p95']:,.2f}")
            mc_cols[4].metric("Probability of Loss", f"{monte_carlo['probability_of_loss']:.2%}")

            sample_paths = monte_carlo["paths"][:, : min(50, monte_carlo["paths"].shape[1])]
            fan_chart = go.Figure()
            for index in range(sample_paths.shape[1]):
                fan_chart.add_trace(go.Scatter(y=sample_paths[:, index], mode="lines", line={"width": 1}, opacity=0.20, showlegend=False))
            fan_chart.update_layout(title="Monte Carlo Fan Chart", template="plotly_white")
            st.plotly_chart(fan_chart, use_container_width=True)

            distribution_chart = go.Figure()
            distribution_chart.add_trace(go.Histogram(x=monte_carlo["final_values"], nbinsx=40))
            distribution_chart.update_layout(title="Distribution of Final Portfolio Values", template="plotly_white")
            st.plotly_chart(distribution_chart, use_container_width=True)
        except Exception as error:
            st.error(f"Monte Carlo simulation failed: {error}")

    st.subheader("Kelly Criterion Position Sizing")
    for _, row in holdings.iterrows():
        kelly = estimate_kelly_position(row["AI_Score"], portfolio_analysis["risk"]["upside_estimate"], portfolio_analysis["risk"]["downside_var"])
        st.write(
            f"{row['Ticker']}: win probability {kelly['win_probability']:.2%}, reward/risk {kelly['reward_risk_ratio']:.2f}, "
            f"Kelly {kelly['kelly_fraction']:.2%}, half-Kelly {kelly['half_kelly']:.2%}, max allocation {kelly['max_allocation']:.2%}"
        )
    st.warning("Kelly sizing is aggressive and should be used conservatively.")


def main() -> None:
    init_session_state()
    st.session_state["watchlists"] = load_watchlists()

    st.title("FinX Smart Suggestion App")
    st.write(
        "Analyze single stocks, live portfolios, screeners, options, forecasting, optimization, "
        "institutional activity, and saved watchlists with free market data and rule-based scoring."
    )
    st.caption(APP_DISCLAIMER)
    st.info("All Indian stock symbols are NSE Yahoo Finance tickers using the .NS suffix.")

    render_authentication_sidebar()
    with st.sidebar:
        st.header("Analysis Controls")
        timeframe_label = st.selectbox("Select timeframe", list(PERIOD_OPTIONS.keys()), index=3)
        selected_period = PERIOD_OPTIONS[timeframe_label]
        refresh_cache_button()
        st.caption("Performance safeguard: bulk screening is capped at 50 tickers by default.")

    tabs = st.tabs(
        [
            "Single Stock Analysis",
            "Portfolio Upload & Live Analysis",
            "Multi-Stock Screener",
            "Options Chain Analysis",
            "Forecasting",
            "Portfolio Optimization",
            "Market Activity",
            "Saved Watchlists",
        ]
    )

    with tabs[0]:
        selected_ticker = st.selectbox("Select Stock", STOCKS)
        try:
            render_single_stock_tab(selected_ticker, selected_period)
        except Exception as error:
            st.error(f"Single-stock analysis failed: {error}")

    with tabs[1]:
        try:
            render_portfolio_tab(selected_period)
        except Exception as error:
            st.error(f"Portfolio analysis failed: {error}")

    with tabs[2]:
        try:
            render_screener_tab()
        except Exception as error:
            st.error(f"Screener failed: {error}")

    with tabs[3]:
        render_options_tab()

    with tabs[4]:
        render_forecasting_tab()

    with tabs[5]:
        render_portfolio_optimization_tab()

    with tabs[6]:
        render_market_activity_tab()

    with tabs[7]:
        render_watchlist_manager()


if __name__ == "__main__":
    main()
