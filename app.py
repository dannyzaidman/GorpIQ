from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from gorpiq.config import DB_PATH, DISCLAIMER_TEXT
from gorpiq.data_provider import YFinanceDataProvider
from gorpiq.database import (
    get_feature_summary,
    get_price_summary,
    initialize_database,
    load_features,
    load_prices,
    upsert_features,
    upsert_prices,
)
from gorpiq.feature_engineering import calculate_features_for_all_tickers
from gorpiq.utils import ensure_data_directories, normalize_tickers


st.set_page_config(page_title="GorpIQ", layout="wide")


def _read_uploaded_watchlist(uploaded_file) -> list[str]:
    if uploaded_file is None:
        return []

    frame = pd.read_csv(uploaded_file)
    if frame.empty:
        return []

    ticker_column = "ticker" if "ticker" in frame.columns else frame.columns[0]
    return normalize_tickers(frame[ticker_column].dropna().astype(str).tolist())


def _render_overview() -> None:
    st.title("GorpIQ")
    st.caption("Local swing-trade research and decision support")

    st.warning(DISCLAIMER_TEXT)

    st.subheader("Current MVP workflow")
    st.markdown(
        """
        1. Enter or upload a watchlist.
        2. Choose a historical date range.
        3. Download daily OHLCV data from Yahoo Finance through yfinance.
        4. Store the normalized data in local SQLite.
        5. Calculate initial historical technical features.
        6. Inspect stored price and feature coverage.
        """
    )

    st.subheader("Local data status")
    summary = get_price_summary()
    if summary.empty:
        st.info("No daily price data has been stored yet.")
    else:
        st.dataframe(summary, width="stretch", hide_index=True)

    st.caption(f"SQLite database: {DB_PATH}")


def _render_watchlist_download() -> None:
    st.title("Watchlist & Data Download")
    st.warning("Data quality depends on yfinance/Yahoo Finance and may be incomplete or adjusted differently than professional datasets.")

    manual_tickers = st.text_area(
        "Tickers",
        value="AAPL, MSFT, NVDA, SPY, QQQ",
        help="Enter tickers separated by commas, spaces, or new lines.",
    )
    uploaded_file = st.file_uploader("Upload watchlist CSV", type=["csv"])

    uploaded_tickers = _read_uploaded_watchlist(uploaded_file)
    tickers = normalize_tickers(manual_tickers) + [t for t in uploaded_tickers if t not in normalize_tickers(manual_tickers)]

    today = date.today()
    default_start = today - timedelta(days=365 * 2)
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start date", value=default_start)
    with col_end:
        end_date = st.date_input("End date", value=today)

    st.write("Normalized watchlist:", ", ".join(tickers) if tickers else "None")

    if st.button("Download and store daily prices", type="primary"):
        if not tickers:
            st.error("Enter at least one ticker.")
        elif start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            provider = YFinanceDataProvider()
            with st.spinner("Downloading daily prices..."):
                prices = provider.download_daily_prices(tickers, start_date, end_date)

            if prices.empty:
                st.error("No price rows were returned for the selected watchlist and date range.")
            else:
                rows_written = upsert_prices(prices)
                st.success(f"Stored {rows_written:,} daily price rows.")
                st.dataframe(prices.tail(200), width="stretch", hide_index=True)

    st.subheader("Stored price coverage")
    summary = get_price_summary()
    if summary.empty:
        st.info("No stored data yet.")
    else:
        st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Stored OHLCV preview")
    stored = load_prices(tickers=tickers or None, start_date=start_date, end_date=end_date)
    if stored.empty:
        st.info("No stored rows match the current filters.")
    else:
        st.dataframe(stored.tail(500), width="stretch", hide_index=True)


def _render_feature_calculation() -> None:
    st.title("Feature Calculation")
    st.warning(
        "Initial MVP features use only each ticker's current and prior daily rows. "
        "ATR uses stored unadjusted high, low, and close because adjusted OHLC fields are not stored yet."
    )

    price_summary = get_price_summary()
    if price_summary.empty:
        st.info("Download daily prices before calculating features.")
        return

    available_tickers = price_summary["ticker"].tolist()
    selected_tickers = st.multiselect("Tickers", available_tickers, default=available_tickers)

    first_available = pd.to_datetime(price_summary["first_date"]).min().date()
    last_available = pd.to_datetime(price_summary["last_date"]).max().date()
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Feature start date", value=first_available, min_value=first_available, max_value=last_available)
    with col_end:
        end_date = st.date_input("Feature end date", value=last_available, min_value=first_available, max_value=last_available)

    if st.button("Calculate initial MVP features", type="primary"):
        if not selected_tickers:
            st.error("Select at least one ticker.")
        elif start_date > end_date:
            st.error("Start date must be on or before end date.")
        else:
            with st.spinner("Calculating features from stored price history..."):
                price_history = load_prices(tickers=selected_tickers, end_date=end_date)
                all_features = calculate_features_for_all_tickers(price_history)
                if all_features.empty:
                    filtered_features = all_features
                else:
                    filtered_features = all_features[
                        (all_features["date"].dt.date >= start_date)
                        & (all_features["date"].dt.date <= end_date)
                    ].reset_index(drop=True)
                rows_written = upsert_features(filtered_features)

            st.success(
                f"Processed {len(selected_tickers):,} tickers and created/updated "
                f"{rows_written:,} feature rows for {start_date} through {end_date}."
            )

    st.subheader("Stored feature coverage")
    feature_summary = get_feature_summary()
    if feature_summary.empty:
        st.info("No stored features yet.")
    else:
        st.dataframe(feature_summary, width="stretch", hide_index=True)

    st.subheader("Feature preview")
    features = load_features(tickers=selected_tickers or None, start_date=start_date, end_date=end_date)
    if features.empty:
        st.info("No stored feature rows match the current filters.")
        return

    st.dataframe(features.tail(500), width="stretch", hide_index=True)

    st.subheader("Missing values by feature")
    feature_value_columns = [column for column in features.columns if column not in {"date", "ticker", "created_at", "updated_at"}]
    missing_counts = (
        features[feature_value_columns]
        .isna()
        .sum()
        .rename("missing_values")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missing_counts = missing_counts[missing_counts["missing_values"] > 0].sort_values(
        ["missing_values", "feature"], ascending=[False, True]
    )
    if missing_counts.empty:
        st.success("No missing values in the current feature view.")
    else:
        st.dataframe(missing_counts, width="stretch", hide_index=True)

    st.download_button(
        "Export features to CSV",
        data=features.to_csv(index=False).encode("utf-8"),
        file_name="gorpiq_features.csv",
        mime="text/csv",
    )


def _render_placeholder_page(title: str, description: str) -> None:
    st.title(title)
    st.info(description)
    st.caption("This module is intentionally left for the next build step after price ingestion is stable.")


def main() -> None:
    ensure_data_directories()
    initialize_database()

    page = st.sidebar.radio(
        "Page",
        [
            "Home / Overview",
            "Watchlist & Data Download",
            "Feature Calculation",
            "Label Generation",
            "Backtest",
            "Current Screener",
            "Trade Tracker",
            "Sell Alerts",
        ],
    )

    if page == "Home / Overview":
        _render_overview()
    elif page == "Watchlist & Data Download":
        _render_watchlist_download()
    elif page == "Feature Calculation":
        _render_feature_calculation()
    elif page == "Label Generation":
        _render_placeholder_page("Label Generation", "Forward-looking labels will be generated only for backtesting and never used as current-day features.")
    elif page == "Backtest":
        _render_placeholder_page("Backtest", "Backtesting will rank historical candidates and compare selected forward returns against SPY.")
    elif page == "Current Screener":
        _render_placeholder_page("Current Screener", "Current ranking will appear after MVP features and transparent scoring are implemented.")
    elif page == "Trade Tracker":
        _render_placeholder_page("Trade Tracker", "Manual trade logging will be added with local SQLite storage.")
    elif page == "Sell Alerts":
        _render_placeholder_page("Sell Alerts", "Sell alerts will evaluate active manual trades against explicit rules.")


if __name__ == "__main__":
    main()
