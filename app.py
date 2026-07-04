import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="FinX Scripts", page_icon="📈", layout="wide")


# Map user-friendly labels to yfinance period values.
PERIOD_OPTIONS = {
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
    "Maximum available data": "max",
}


@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker_symbol: str, period: str) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance."""
    cleaned_ticker = ticker_symbol.strip().upper()

    if not cleaned_ticker:
        raise ValueError("Please enter a stock ticker symbol.")

    ticker = yf.Ticker(cleaned_ticker)
    history = ticker.history(period=period, auto_adjust=False)

    if history.empty:
        raise ValueError(
            "No data was returned for this ticker and time period. "
            "Please check the symbol and try again."
        )

    history = history.reset_index()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
    return history


def build_price_chart(dataframe: pd.DataFrame, ticker_symbol: str) -> go.Figure:
    """Create an interactive Plotly line chart for closing prices."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dataframe["Date"],
            y=dataframe["Close"],
            mode="lines",
            name="Close Price",
            line={"width": 3, "color": "#0F766E"},
            hovertemplate="Date: %{x|%d %b %Y}<br>Close: %{y:.2f}<extra></extra>",
        )
    )

    figure.update_layout(
        title=f"{ticker_symbol} Closing Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


# Main app header and description.
st.title("FinX Scripts")
st.write(
    "Track historical stock closing prices with Yahoo Finance data. "
    "Enter a ticker, choose a time range, and explore the chart instantly."
)


# Sidebar controls make the layout beginner-friendly and tidy.
st.sidebar.header("Stock Settings")
ticker_input = st.sidebar.text_input(
    "Enter a stock ticker",
    value="AAPL",
    help="Examples: RELIANCE.NS, TCS.NS, AAPL, MSFT",
)
selected_label = st.sidebar.selectbox(
    "Select a time period",
    options=list(PERIOD_OPTIONS.keys()),
    index=3,
)
selected_period = PERIOD_OPTIONS[selected_label]


try:
    stock_data = fetch_stock_data(ticker_input, selected_period)

    latest_close = stock_data["Close"].iloc[-1]
    start_date = stock_data["Date"].min().date()
    end_date = stock_data["Date"].max().date()

    metric_column, range_column = st.columns(2)
    metric_column.metric("Latest Close Price", f"{latest_close:,.2f}")
    range_column.metric("Date Range", f"{start_date} to {end_date}")

    st.plotly_chart(
        build_price_chart(stock_data, ticker_input.strip().upper()),
        use_container_width=True,
    )

    st.subheader("Data Preview")
    preview_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    st.dataframe(stock_data[preview_columns].tail(10), use_container_width=True)

except ValueError as error:
    st.warning(str(error))
except Exception:
    st.error(
        "Something went wrong while fetching stock data. "
        "Please try again in a moment."
    )
