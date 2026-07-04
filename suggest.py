from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

SAMPLE_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "AAPL",
    "MSFT",
]

REQUIRED_PORTFOLIO_COLUMNS = ["Ticker", "Quantity", "Average_Buy_Price"]


# Data loading helpers
@st.cache_data(show_spinner=False, ttl=900)
def fetch_stock_data(ticker_symbol: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance and normalize the index."""
    cleaned_ticker = str(ticker_symbol).strip().upper()
    if not cleaned_ticker:
        raise ValueError("Please enter a valid stock ticker.")

    history = yf.Ticker(cleaned_ticker).history(period=period, auto_adjust=False)
    if history.empty:
        raise ValueError(
            f"No price data was returned for {cleaned_ticker}. "
            "Please check the ticker or try another timeframe."
        )

    history = history.reset_index()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
    history = history.dropna(subset=["Open", "High", "Low", "Close"])
    if history.empty:
        raise ValueError(f"{cleaned_ticker} returned empty OHLC data after cleanup.")

    return history


@st.cache_data(show_spinner=False, ttl=900)
def fetch_fundamentals(ticker_symbol: str) -> dict[str, Any]:
    """Fetch fundamental metadata and safely return only the fields we use."""
    info = yf.Ticker(str(ticker_symbol).strip().upper()).info or {}
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
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
    }


# Indicator calculations
def calculate_pivot_levels(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate classic pivot levels from the latest OHLC row."""
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
            "Value": [
                pivot_point,
                resistance_1,
                resistance_2,
                resistance_3,
                support_1,
                support_2,
                support_3,
            ],
        }
    )


def calculate_rsi(dataframe: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate RSI using the standard rolling average approach."""
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
    """Calculate MACD, signal line, and histogram."""
    result = dataframe.copy()
    fast_ema = result["Close"].ewm(span=12, adjust=False).mean()
    slow_ema = result["Close"].ewm(span=26, adjust=False).mean()
    result["MACD"] = fast_ema - slow_ema
    result["MACD_Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["MACD_Histogram"] = result["MACD"] - result["MACD_Signal"]
    return result


def calculate_mean_reversion(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling mean, rolling standard deviation, and z-score."""
    result = dataframe.copy()
    result["Rolling_Mean_20"] = result["Close"].rolling(window=20, min_periods=20).mean()
    result["Rolling_Std_20"] = result["Close"].rolling(window=20, min_periods=20).std()
    result["Z_Score"] = (
        (result["Close"] - result["Rolling_Mean_20"])
        / result["Rolling_Std_20"].replace(0, np.nan)
    )
    result["Z_Score"] = result["Z_Score"].replace([np.inf, -np.inf], np.nan).fillna(0)
    result["SMA_50"] = result["Close"].rolling(window=50, min_periods=1).mean()
    return result


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
    if difference > 0:
        return "Bullish"
    return "Bearish"


def interpret_mean_reversion(z_score: float) -> str:
    if z_score > 2:
        return "Overbought / possible mean reversal downside"
    if z_score < -2:
        return "Oversold / possible mean reversal upside"
    return "Neutral"


# Scoring helpers
def add_scored_reason(
    bucket: list[dict[str, Any]],
    label: str,
    score: float,
    reason: str,
) -> None:
    bucket.append({"Factor": label, "Score": round(score, 2), "Reason": reason})


def safe_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)) and not pd.isna(value):
        return float(value)
    return None


def calculate_ai_smartness_score(
    dataframe: pd.DataFrame,
    fundamentals: dict[str, Any],
    pivot_levels: pd.DataFrame,
) -> dict[str, Any]:
    """Blend technical, fundamental, and risk signals into a transparent score."""
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
    key_reasons: list[str] = []

    rsi_condition = interpret_rsi(latest_rsi)
    if rsi_condition == "Oversold":
        add_scored_reason(technical_details, "RSI", 18, f"RSI at {latest_rsi:.2f} suggests oversold conditions.")
        key_reasons.append("RSI is in oversold territory.")
    elif rsi_condition == "Neutral":
        add_scored_reason(technical_details, "RSI", 10, f"RSI at {latest_rsi:.2f} is neutral.")
    else:
        add_scored_reason(technical_details, "RSI", 4, f"RSI at {latest_rsi:.2f} suggests overbought conditions.")
        key_reasons.append("RSI is elevated and may limit upside.")

    macd_condition = interpret_macd(latest_macd, latest_signal)
    if macd_condition == "Bullish":
        add_scored_reason(
            technical_details,
            "MACD",
            18,
            f"MACD ({latest_macd:.2f}) is above signal ({latest_signal:.2f}).",
        )
        key_reasons.append("MACD momentum is bullish.")
    elif macd_condition == "Neutral":
        add_scored_reason(
            technical_details,
            "MACD",
            10,
            f"MACD ({latest_macd:.2f}) is close to signal ({latest_signal:.2f}).",
        )
    else:
        add_scored_reason(
            technical_details,
            "MACD",
            4,
            f"MACD ({latest_macd:.2f}) is below signal ({latest_signal:.2f}).",
        )
        key_reasons.append("MACD momentum is bearish.")

    mean_reversion_condition = interpret_mean_reversion(latest_z_score)
    if mean_reversion_condition.startswith("Oversold"):
        add_scored_reason(
            technical_details,
            "Mean Reversion",
            16,
            f"Z-score at {latest_z_score:.2f} suggests oversold conditions.",
        )
        key_reasons.append("Price is stretched below its 20-day mean.")
    elif mean_reversion_condition == "Neutral":
        add_scored_reason(
            technical_details,
            "Mean Reversion",
            10,
            f"Z-score at {latest_z_score:.2f} is neutral.",
        )
    else:
        add_scored_reason(
            technical_details,
            "Mean Reversion",
            4,
            f"Z-score at {latest_z_score:.2f} suggests overbought conditions.",
        )
        key_reasons.append("Price is stretched above its 20-day mean.")

    trend_ratio = latest_close / latest_sma50 if latest_sma50 else 1
    if trend_ratio > 1.03:
        add_scored_reason(
            technical_details,
            "Trend vs SMA50",
            16,
            f"Price is {((trend_ratio - 1) * 100):.2f}% above the 50-day average.",
        )
        key_reasons.append("Price remains above the 50-day trend.")
    elif trend_ratio >= 0.97:
        add_scored_reason(
            technical_details,
            "Trend vs SMA50",
            10,
            "Price is trading close to its 50-day average.",
        )
    else:
        add_scored_reason(
            technical_details,
            "Trend vs SMA50",
            4,
            f"Price is {((1 - trend_ratio) * 100):.2f}% below the 50-day average.",
        )
        key_reasons.append("Price is below the 50-day trend.")

    pivot_map = dict(zip(pivot_levels["Level"], pivot_levels["Value"]))
    support_1 = float(pivot_map["S1"])
    resistance_1 = float(pivot_map["R1"])
    distance_to_support = ((latest_close - support_1) / latest_close) * 100
    distance_to_resistance = ((resistance_1 - latest_close) / latest_close) * 100
    if distance_to_support <= 2 and distance_to_support >= 0:
        add_scored_reason(
            technical_details,
            "Support/Resistance",
            16,
            "Price is trading near support, which may improve reward-to-risk.",
        )
        key_reasons.append("Price is near first support.")
    elif distance_to_resistance <= 2 and distance_to_resistance >= 0:
        add_scored_reason(
            technical_details,
            "Support/Resistance",
            5,
            "Price is close to resistance, which may cap upside in the short term.",
        )
        key_reasons.append("Price is near first resistance.")
    else:
        add_scored_reason(
            technical_details,
            "Support/Resistance",
            10,
            "Price is between support and resistance without an extreme setup.",
        )

    trailing_pe = safe_number(fundamentals.get("trailingPE"))
    if trailing_pe is None:
        add_scored_reason(fundamental_details, "Trailing P/E", 7, "Trailing P/E is unavailable.")
    elif 0 < trailing_pe <= 20:
        add_scored_reason(fundamental_details, "Trailing P/E", 14, f"Trailing P/E of {trailing_pe:.2f} looks reasonable.")
    elif trailing_pe <= 35:
        add_scored_reason(fundamental_details, "Trailing P/E", 10, f"Trailing P/E of {trailing_pe:.2f} is acceptable.")
    else:
        add_scored_reason(fundamental_details, "Trailing P/E", 4, f"Trailing P/E of {trailing_pe:.2f} looks expensive.")
        key_reasons.append("Valuation looks rich on trailing earnings.")

    price_to_book = safe_number(fundamentals.get("priceToBook"))
    if price_to_book is None:
        add_scored_reason(fundamental_details, "Price/Book", 7, "Price-to-book ratio is unavailable.")
    elif price_to_book <= 3:
        add_scored_reason(fundamental_details, "Price/Book", 14, f"Price-to-book of {price_to_book:.2f} looks healthy.")
    elif price_to_book <= 6:
        add_scored_reason(fundamental_details, "Price/Book", 10, f"Price-to-book of {price_to_book:.2f} is moderate.")
    else:
        add_scored_reason(fundamental_details, "Price/Book", 4, f"Price-to-book of {price_to_book:.2f} is elevated.")

    debt_to_equity = safe_number(fundamentals.get("debtToEquity"))
    if debt_to_equity is None:
        add_scored_reason(fundamental_details, "Debt/Equity", 7, "Debt-to-equity is unavailable.")
    elif debt_to_equity <= 50:
        add_scored_reason(fundamental_details, "Debt/Equity", 14, f"Debt-to-equity of {debt_to_equity:.2f} is conservative.")
    elif debt_to_equity <= 120:
        add_scored_reason(fundamental_details, "Debt/Equity", 9, f"Debt-to-equity of {debt_to_equity:.2f} is manageable.")
    else:
        add_scored_reason(fundamental_details, "Debt/Equity", 3, f"Debt-to-equity of {debt_to_equity:.2f} is high.")
        key_reasons.append("Balance-sheet leverage is relatively high.")

    profit_margin = safe_number(fundamentals.get("profitMargins"))
    if profit_margin is None:
        add_scored_reason(fundamental_details, "Profit Margin", 7, "Profit margin is unavailable.")
    elif profit_margin >= 0.20:
        add_scored_reason(fundamental_details, "Profit Margin", 14, f"Profit margin of {profit_margin:.2%} is strong.")
        key_reasons.append("Profitability is strong.")
    elif profit_margin >= 0.08:
        add_scored_reason(fundamental_details, "Profit Margin", 10, f"Profit margin of {profit_margin:.2%} is decent.")
    else:
        add_scored_reason(fundamental_details, "Profit Margin", 4, f"Profit margin of {profit_margin:.2%} is thin.")

    revenue_growth = safe_number(fundamentals.get("revenueGrowth"))
    if revenue_growth is None:
        add_scored_reason(fundamental_details, "Revenue Growth", 7, "Revenue growth is unavailable.")
    elif revenue_growth >= 0.15:
        add_scored_reason(fundamental_details, "Revenue Growth", 14, f"Revenue growth of {revenue_growth:.2%} is strong.")
        key_reasons.append("Revenue growth is healthy.")
    elif revenue_growth >= 0:
        add_scored_reason(fundamental_details, "Revenue Growth", 10, f"Revenue growth of {revenue_growth:.2%} is positive.")
    else:
        add_scored_reason(fundamental_details, "Revenue Growth", 3, f"Revenue growth of {revenue_growth:.2%} is negative.")

    return_on_equity = safe_number(fundamentals.get("returnOnEquity"))
    if return_on_equity is None:
        add_scored_reason(fundamental_details, "ROE", 7, "Return on equity is unavailable.")
    elif return_on_equity >= 0.15:
        add_scored_reason(fundamental_details, "ROE", 14, f"ROE of {return_on_equity:.2%} is strong.")
    elif return_on_equity >= 0.08:
        add_scored_reason(fundamental_details, "ROE", 10, f"ROE of {return_on_equity:.2%} is acceptable.")
    else:
        add_scored_reason(fundamental_details, "ROE", 4, f"ROE of {return_on_equity:.2%} is weak.")

    daily_returns = dataframe["Close"].pct_change().dropna()
    annualized_volatility = float(daily_returns.std() * np.sqrt(252)) if not daily_returns.empty else 0
    running_max = dataframe["Close"].cummax()
    drawdown_series = (dataframe["Close"] / running_max) - 1
    max_drawdown = float(drawdown_series.min()) if not drawdown_series.empty else 0

    if annualized_volatility <= 0.20:
        add_scored_reason(risk_details, "Volatility", 12, f"Annualized volatility of {annualized_volatility:.2%} is contained.")
    elif annualized_volatility <= 0.35:
        add_scored_reason(risk_details, "Volatility", 8, f"Annualized volatility of {annualized_volatility:.2%} is moderate.")
    else:
        add_scored_reason(risk_details, "Volatility", 3, f"Annualized volatility of {annualized_volatility:.2%} is high.")
        key_reasons.append("Volatility is elevated.")

    if max_drawdown >= -0.10:
        add_scored_reason(risk_details, "Drawdown", 12, f"Maximum drawdown of {max_drawdown:.2%} has been limited.")
    elif max_drawdown >= -0.25:
        add_scored_reason(risk_details, "Drawdown", 8, f"Maximum drawdown of {max_drawdown:.2%} is manageable.")
    else:
        add_scored_reason(risk_details, "Drawdown", 3, f"Maximum drawdown of {max_drawdown:.2%} is deep.")
        key_reasons.append("Historical drawdown is meaningful.")

    if distance_to_support <= 2 and distance_to_support >= 0:
        add_scored_reason(risk_details, "Distance from Support", 8, f"Only {distance_to_support:.2f}% above support.")
    elif distance_to_support <= 5:
        add_scored_reason(risk_details, "Distance from Support", 6, f"{distance_to_support:.2f}% above support.")
    else:
        add_scored_reason(risk_details, "Distance from Support", 3, f"{distance_to_support:.2f}% above support.")

    if distance_to_resistance >= 5:
        add_scored_reason(risk_details, "Distance from Resistance", 8, f"{distance_to_resistance:.2f}% room before resistance.")
    elif distance_to_resistance >= 2:
        add_scored_reason(risk_details, "Distance from Resistance", 6, f"{distance_to_resistance:.2f}% room before resistance.")
    else:
        add_scored_reason(risk_details, "Distance from Resistance", 3, f"Only {distance_to_resistance:.2f}% room before resistance.")

    technical_score = min(40, sum(item["Score"] for item in technical_details))
    fundamental_score = min(35, sum(item["Score"] for item in fundamental_details))
    risk_score = min(25, sum(item["Score"] for item in risk_details))
    final_score = int(round(min(100, technical_score + fundamental_score + risk_score)))

    if final_score >= 70:
        recommendation = "Buy"
    elif final_score >= 40:
        recommendation = "Hold"
    else:
        recommendation = "Sell"

    if not key_reasons:
        key_reasons.append("Signals are mixed, so the score remains balanced.")

    return {
        "score": final_score,
        "recommendation": recommendation,
        "technical_score": round(technical_score, 2),
        "fundamental_score": round(fundamental_score, 2),
        "risk_score": round(risk_score, 2),
        "technical_breakdown": pd.DataFrame(technical_details),
        "fundamental_breakdown": pd.DataFrame(fundamental_details),
        "risk_breakdown": pd.DataFrame(risk_details),
        "key_reasons": key_reasons[:6],
        "rsi_condition": rsi_condition,
        "macd_condition": macd_condition,
        "mean_reversion_condition": mean_reversion_condition,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "distance_to_support_pct": distance_to_support,
        "distance_to_resistance_pct": distance_to_resistance,
    }


# Visualization helpers
def build_price_chart(dataframe: pd.DataFrame, ticker_symbol: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dataframe["Date"],
            y=dataframe["Close"],
            mode="lines",
            name="Close",
            line={"color": "#0b6e4f", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dataframe["Date"],
            y=dataframe["SMA_50"],
            mode="lines",
            name="SMA 50",
            line={"color": "#f4a259", "width": 2, "dash": "dash"},
        )
    )
    figure.update_layout(
        title=f"{ticker_symbol} Closing Price",
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


def build_rsi_chart(dataframe: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dataframe["Date"],
            y=dataframe["RSI"],
            mode="lines",
            name="RSI",
            line={"color": "#005f73", "width": 2},
        )
    )
    figure.add_hline(y=70, line_dash="dash", line_color="#c1121f")
    figure.add_hline(y=30, line_dash="dash", line_color="#2a9d8f")
    figure.update_layout(
        title="RSI (14)",
        template="plotly_white",
        yaxis_title="RSI",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


def build_macd_chart(dataframe: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": False}]])
    figure.add_trace(
        go.Bar(
            x=dataframe["Date"],
            y=dataframe["MACD_Histogram"],
            name="Histogram",
            marker_color="#94d2bd",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dataframe["Date"],
            y=dataframe["MACD"],
            mode="lines",
            name="MACD",
            line={"color": "#0a9396", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dataframe["Date"],
            y=dataframe["MACD_Signal"],
            mode="lines",
            name="Signal",
            line={"color": "#ee9b00", "width": 2},
        )
    )
    figure.update_layout(
        title="MACD",
        template="plotly_white",
        barmode="relative",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


def format_large_number(value: Any) -> str:
    numeric_value = safe_number(value)
    if numeric_value is None:
        return "Not available"
    return f"{numeric_value:,.2f}"


def format_ratio(value: Any) -> str:
    numeric_value = safe_number(value)
    if numeric_value is None:
        return "Not available"
    return f"{numeric_value:.2%}"


def format_float(value: Any) -> str:
    numeric_value = safe_number(value)
    if numeric_value is None:
        return "Not available"
    return f"{numeric_value:.2f}"


def display_fundamentals(fundamentals: dict[str, Any]) -> None:
    """Display fundamentals in a tidy table."""
    table = pd.DataFrame(
        {
            "Metric": [
                "Market Cap",
                "Trailing P/E",
                "Forward P/E",
                "Price to Book",
                "Dividend Yield",
                "Return on Equity",
                "Debt to Equity",
                "Revenue Growth",
                "Profit Margins",
                "Sector",
                "Industry",
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
                fundamentals.get("sector") or "Not available",
                fundamentals.get("industry") or "Not available",
            ],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def calculate_portfolio_var(
    price_history: pd.DataFrame,
    holdings: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate weighted portfolio return series, volatility, VaR, and simple upside/downside estimates."""
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


def create_portfolio_suggestions(
    holdings: pd.DataFrame,
    portfolio_risk: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Turn stock-level scores and portfolio concentration into practical suggestions."""
    suggestions = []
    portfolio_messages: list[str] = []

    for _, row in holdings.iterrows():
        if row["AI_Score"] >= 70:
            action = "Increase / accumulate"
            reason = "High score with supportive technical and/or fundamental signals."
        elif row["AI_Score"] >= 40:
            action = "Hold / monitor"
            reason = "Balanced score with mixed signals."
        else:
            action = "Reduce / review"
            reason = "Low score reflects weaker momentum, valuation, or risk profile."

        if row["Weight"] > 0.35:
            reason += " Portfolio weight is high, so concentration risk deserves attention."

        suggestions.append(
            {
                "Ticker": row["Ticker"],
                "AI Score": row["AI_Score"],
                "Recommendation": row["Recommendation"],
                "Suggested Action": action,
                "Reason": reason,
            }
        )

    if (holdings["Weight"] > 0.35).any():
        portfolio_messages.append("One holding exceeds 35% of the portfolio, which increases concentration risk.")
    if portfolio_risk["volatility"] > 0.30:
        portfolio_messages.append("Portfolio volatility is high relative to a diversified long-term allocation.")
    if portfolio_risk["var_95"] < -0.03:
        portfolio_messages.append("Historical 95% daily VaR is elevated, so downside protection may be worth reviewing.")
    if not portfolio_messages:
        portfolio_messages.append("Portfolio risk looks moderate based on current holdings and historical returns.")

    return pd.DataFrame(suggestions), portfolio_messages


# Analysis views
def analyze_single_stock(ticker_symbol: str, period: str) -> None:
    """Run the full stock analysis and render the Streamlit layout."""
    history = fetch_stock_data(ticker_symbol, period)
    history = calculate_rsi(history)
    history = calculate_macd(history)
    history = calculate_mean_reversion(history)

    fundamentals = fetch_fundamentals(ticker_symbol)
    pivot_levels = calculate_pivot_levels(history)
    score_card = calculate_ai_smartness_score(history, fundamentals, pivot_levels)

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
    summary_columns[4].metric("Range", f"{start_date} to {end_date}")

    score_columns = st.columns(4)
    score_columns[0].metric("AI Smartness Score", f"{score_card['score']}/100")
    score_columns[1].metric("Recommendation", score_card["recommendation"])
    score_columns[2].metric("Technical Score", f"{score_card['technical_score']:.1f}/40")
    score_columns[3].metric("Risk Score", f"{score_card['risk_score']:.1f}/25")

    st.plotly_chart(build_price_chart(history, ticker_symbol), use_container_width=True)

    indicator_left, indicator_right = st.columns(2)
    indicator_left.plotly_chart(build_rsi_chart(history), use_container_width=True)
    indicator_right.plotly_chart(build_macd_chart(history), use_container_width=True)

    status_left, status_mid, status_right = st.columns(3)
    status_left.metric("RSI", f"{latest_row['RSI']:.2f}", score_card["rsi_condition"])
    status_mid.metric("MACD", f"{latest_row['MACD']:.2f}", score_card["macd_condition"])
    status_right.metric("Z-Score", f"{latest_row['Z_Score']:.2f}", score_card["mean_reversion_condition"])

    extra_left, extra_mid, extra_right = st.columns(3)
    extra_left.metric("Signal Line", f"{latest_row['MACD_Signal']:.2f}")
    extra_mid.metric("Histogram", f"{latest_row['MACD_Histogram']:.2f}")
    extra_right.metric("Rolling Mean 20", f"{latest_row['Rolling_Mean_20']:.2f}")

    st.subheader("Pivot Table Support and Resistance")
    formatted_pivots = pivot_levels.copy()
    formatted_pivots["Value"] = formatted_pivots["Value"].map(lambda value: f"{value:,.2f}")
    st.dataframe(formatted_pivots, use_container_width=True, hide_index=True)

    st.subheader("Fundamental Analysis")
    display_fundamentals(fundamentals)

    st.subheader("Recommendation Summary")
    for reason in score_card["key_reasons"]:
        st.write(f"- {reason}")

    breakdown_tabs = st.tabs(["Technical Breakdown", "Fundamental Breakdown", "Risk Breakdown"])
    breakdown_tabs[0].dataframe(score_card["technical_breakdown"], use_container_width=True, hide_index=True)
    breakdown_tabs[1].dataframe(score_card["fundamental_breakdown"], use_container_width=True, hide_index=True)
    breakdown_tabs[2].dataframe(score_card["risk_breakdown"], use_container_width=True, hide_index=True)

    with st.expander("Show recent price data"):
        preview_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MACD_Signal", "Z_Score"]
        st.dataframe(history[preview_columns].tail(20), use_container_width=True)


def analyze_portfolio(portfolio_file: Any, period: str) -> None:
    """Validate the uploaded CSV, analyze each holding, and show portfolio metrics."""
    if portfolio_file is None:
        st.info("Upload a CSV file to analyze your portfolio.")
        return

    portfolio_df = pd.read_csv(portfolio_file)
    missing_columns = [column for column in REQUIRED_PORTFOLIO_COLUMNS if column not in portfolio_df.columns]
    if missing_columns:
        raise ValueError(
            "Portfolio CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    working_df = portfolio_df[REQUIRED_PORTFOLIO_COLUMNS].copy()
    working_df["Ticker"] = working_df["Ticker"].astype(str).str.strip().str.upper()
    working_df["Quantity"] = pd.to_numeric(working_df["Quantity"], errors="coerce")
    working_df["Average_Buy_Price"] = pd.to_numeric(working_df["Average_Buy_Price"], errors="coerce")

    if working_df["Ticker"].eq("").any():
        raise ValueError("Portfolio CSV contains an empty ticker value.")
    if working_df["Quantity"].isna().any() or working_df["Average_Buy_Price"].isna().any():
        raise ValueError("Quantity and Average_Buy_Price must be numeric for every row.")
    if (working_df["Quantity"] <= 0).any():
        raise ValueError("Quantity must be greater than zero for every holding.")
    if (working_df["Average_Buy_Price"] <= 0).any():
        raise ValueError("Average_Buy_Price must be greater than zero for every holding.")

    holdings_rows = []
    price_series_map: dict[str, pd.Series] = {}
    issues: list[str] = []

    for _, row in working_df.iterrows():
        ticker = row["Ticker"]
        try:
            history = fetch_stock_data(ticker, period)
            history = calculate_rsi(history)
            history = calculate_macd(history)
            history = calculate_mean_reversion(history)
            fundamentals = fetch_fundamentals(ticker)
            pivot_levels = calculate_pivot_levels(history)
            score_card = calculate_ai_smartness_score(history, fundamentals, pivot_levels)

            latest_close = float(history["Close"].iloc[-1])
            invested_value = float(row["Quantity"] * row["Average_Buy_Price"])
            current_value = float(row["Quantity"] * latest_close)
            profit_loss = current_value - invested_value
            return_pct = (profit_loss / invested_value) * 100 if invested_value else 0

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
                    "Key_Reason": " | ".join(score_card["key_reasons"][:2]),
                }
            )

            close_series = history.set_index("Date")["Close"].rename(ticker)
            price_series_map[ticker] = close_series
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
        (summary["Total Profit / Loss"] / summary["Total Invested Value"]) * 100
        if summary["Total Invested Value"]
        else 0
    )

    price_history = pd.concat(price_series_map.values(), axis=1).sort_index().ffill().dropna(how="all")
    portfolio_risk = calculate_portfolio_var(price_history, holdings)
    suggestions_df, portfolio_messages = create_portfolio_suggestions(holdings, portfolio_risk)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Invested", f"{summary['Total Invested Value']:,.2f}")
    metric_columns[1].metric("Current Value", f"{summary['Current Portfolio Value']:,.2f}")
    metric_columns[2].metric("Total P/L", f"{summary['Total Profit / Loss']:,.2f}")
    metric_columns[3].metric("Return %", f"{summary['Total Return Percentage']:,.2f}%")

    risk_columns = st.columns(4)
    risk_columns[0].metric("Portfolio Volatility", f"{portfolio_risk['volatility']:.2%}")
    risk_columns[1].metric("95% Historical VaR", f"{portfolio_risk['var_95']:.2%}")
    risk_columns[2].metric("Expected Downside VaR", f"{portfolio_risk['downside_var']:.2%}")
    risk_columns[3].metric("Upside Estimate", f"{portfolio_risk['upside_estimate']:.2%}")

    st.subheader("Portfolio Summary Table")
    portfolio_view = holdings.copy()
    portfolio_view["Weight"] = portfolio_view["Weight"].map(lambda value: f"{value:.2%}")
    portfolio_view["Return_Pct"] = portfolio_view["Return_Pct"].map(lambda value: f"{value:.2f}%")
    st.dataframe(portfolio_view, use_container_width=True, hide_index=True)

    st.subheader("Individual Stock Recommendations")
    st.dataframe(suggestions_df, use_container_width=True, hide_index=True)

    st.subheader("Portfolio-Level Recommendation")
    for message in portfolio_messages:
        st.write(f"- {message}")

    returns_chart = go.Figure()
    returns_chart.add_trace(
        go.Scatter(
            x=portfolio_risk["daily_returns"].index,
            y=portfolio_risk["daily_returns"].cumsum(),
            mode="lines",
            name="Cumulative Daily Return",
            line={"color": "#386641", "width": 3},
        )
    )
    returns_chart.update_layout(
        title="Portfolio Cumulative Daily Return",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(returns_chart, use_container_width=True)

    if issues:
        st.warning("Some holdings could not be analyzed:")
        for issue in issues:
            st.write(f"- {issue}")


# App shell
st.title("FinX Smart Suggestion App")
st.write(
    "Analyze individual stocks and uploaded portfolios with Yahoo Finance data, "
    "technical indicators, fundamental checks, and a rule-based recommendation engine."
)
st.caption(
    "Disclaimer: This app is for educational and analytical purposes only. "
    "Its AI Smartness Score is rule-based and is not financial advice."
)

with st.sidebar:
    st.header("Analysis Controls")
    ticker_input = st.text_input("Enter ticker", value="RELIANCE.NS")
    quick_ticker = st.selectbox("Or choose a sample ticker", SAMPLE_TICKERS, index=0)
    timeframe_label = st.selectbox("Select timeframe", list(PERIOD_OPTIONS.keys()), index=3)
    selected_period = PERIOD_OPTIONS[timeframe_label]
    use_sample = st.checkbox("Use selected sample ticker", value=False)
    active_ticker = quick_ticker if use_sample else ticker_input.strip().upper()

tab_single, tab_portfolio = st.tabs(
    ["Single Stock Analysis", "Portfolio Upload & Live Analysis"]
)

with tab_single:
    try:
        analyze_single_stock(active_ticker, selected_period)
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(
            "Something went wrong while analyzing the selected stock. "
            "Please verify the ticker and try again."
        )
        st.caption(f"Technical details: {error}")

with tab_portfolio:
    st.write("Upload a CSV with `Ticker`, `Quantity`, and `Average_Buy_Price` columns.")
    uploaded_file = st.file_uploader("Upload portfolio CSV", type=["csv"])
    try:
        analyze_portfolio(uploaded_file, selected_period)
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(
            "Something went wrong while analyzing the portfolio. "
            "Please review the CSV data and try again."
        )
        st.caption(f"Technical details: {error}")
