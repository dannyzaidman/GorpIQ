from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from gorpiq.config import DB_PATH, DISCLAIMER_TEXT
from gorpiq.data_provider import YFinanceDataProvider
from gorpiq.database import (
    get_feature_summary,
    get_label_summary,
    get_price_summary,
    initialize_database,
    load_features,
    load_labels,
    load_prices,
    upsert_features,
    upsert_labels,
    upsert_prices,
)
from gorpiq.feature_engineering import calculate_features_for_all_tickers
from gorpiq.label_generator import calculate_labels_for_all_tickers
from gorpiq.utils import combine_ticker_sources, ensure_data_directories, normalize_tickers


st.set_page_config(page_title="GorpIQ", layout="wide")


def _read_uploaded_watchlist(uploaded_file) -> list[str]:
    if uploaded_file is None:
        return []

    frame = pd.read_csv(uploaded_file)
    if frame.empty:
        return []

    ticker_column = "ticker" if "ticker" in frame.columns else frame.columns[0]
    return normalize_tickers(frame[ticker_column].dropna().astype(str).tolist())


def _recent_price_rows_by_ticker(prices: pd.DataFrame, rows_per_ticker: int = 20) -> pd.DataFrame:
    """Return recent price rows for each ticker, sorted for cross-ticker scanning."""
    if prices.empty:
        return prices

    recent = (
        prices.sort_values(["ticker", "date"])
        .groupby("ticker", group_keys=False)
        .tail(rows_per_ticker)
        .sort_values(["date", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return recent


def _render_price_row_counts(prices: pd.DataFrame, expected_tickers: list[str]) -> None:
    if prices.empty:
        return

    counts = prices.groupby("ticker").size().rename("rows").reset_index()
    st.dataframe(counts, width="stretch", hide_index=True)

    returned_tickers = set(counts["ticker"])
    missing_tickers = [ticker for ticker in expected_tickers if ticker not in returned_tickers]
    if missing_tickers:
        st.warning(
            "No price rows were returned for: "
            + ", ".join(missing_tickers)
            + ". This can happen if Yahoo Finance does not recognize a symbol or has no data for the selected dates."
        )


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

    st.subheader("Manual Watchlist")
    manual_ticker_text = st.text_area(
        "Ticker symbols",
        value="",
        key="manual_ticker_text",
        placeholder="JPM\nAAPL, MSFT, NVDA\nSPY QQQ",
        help="Type one ticker or many tickers. Separate them with commas, spaces, tabs, or new lines.",
        height=110,
    )

    st.subheader("CSV Watchlist Upload")
    st.caption("Optional. Upload a CSV with a `ticker` column, or use the first column as ticker symbols.")
    uploaded_file = st.file_uploader(
        "Choose watchlist CSV",
        type=["csv"],
        key="watchlist_csv_upload",
    )

    uploaded_tickers = _read_uploaded_watchlist(uploaded_file)
    tickers = combine_ticker_sources(manual_ticker_text, uploaded_tickers)

    st.subheader("Parsed Watchlist")
    if tickers:
        st.success(f"{len(tickers):,} tickers parsed. Click the download button below to fetch price data.")
        st.code(", ".join(tickers), language=None)
    else:
        st.warning("Enter tickers manually or upload a CSV watchlist before downloading data.")

    today = date.today()
    default_start = today - timedelta(days=365 * 2)
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start date", value=default_start)
    with col_end:
        end_date = st.date_input("End date", value=today)

    if st.button("Download and store daily prices", type="primary"):
        if not tickers:
            st.error("Enter at least one ticker manually or upload a CSV watchlist.")
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
                st.subheader("Downloaded rows by ticker")
                _render_price_row_counts(prices, tickers)
                st.subheader("Downloaded OHLCV preview")
                st.caption("Showing recent rows for each ticker, not just the last ticker alphabetically.")
                st.dataframe(_recent_price_rows_by_ticker(prices), width="stretch", hide_index=True)

    st.subheader("Stored price coverage")
    summary = get_price_summary()
    if summary.empty:
        st.info("No stored data yet.")
    else:
        st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Stored OHLCV preview")
    if not tickers:
        st.info("Parsed watchlist rows will appear here after you enter or upload tickers.")
    else:
        stored = load_prices(tickers=tickers, start_date=start_date, end_date=end_date)
        if stored.empty:
            st.info(
                "The parsed tickers are ready, but no stored OHLCV rows match this watchlist and date range yet. "
                "Click “Download and store daily prices” to fetch them."
            )
        else:
            st.caption("Showing recent stored rows for each ticker in the parsed watchlist.")
            _render_price_row_counts(stored, tickers)
            st.dataframe(_recent_price_rows_by_ticker(stored), width="stretch", hide_index=True)


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


def _render_label_generation() -> None:
    st.title("Label Generation")
    st.warning(
        "Forward-looking labels use future adjusted-close data. They are historical outcomes for backtesting only "
        "and must never be used as current-day input features or screener signals."
    )

    price_summary = get_price_summary()
    if price_summary.empty:
        st.info("Download daily prices before generating labels.")
        return

    available_tickers = price_summary["ticker"].tolist()
    selected_tickers = st.multiselect("Tickers", available_tickers, default=available_tickers, key="label_tickers")

    first_available = pd.to_datetime(price_summary["first_date"]).min().date()
    last_available = pd.to_datetime(price_summary["last_date"]).max().date()
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Label start date",
            value=first_available,
            min_value=first_available,
            max_value=last_available,
            key="label_start_date",
        )
    with col_end:
        end_date = st.date_input(
            "Label end date",
            value=last_available,
            min_value=first_available,
            max_value=last_available,
            key="label_end_date",
        )

    if st.button("Generate forward-looking labels", type="primary"):
        if not selected_tickers:
            st.error("Select at least one ticker.")
        elif start_date > end_date:
            st.error("Start date must be on or before end date.")
        else:
            with st.spinner("Generating forward-looking labels from stored adjusted-close history..."):
                price_history = load_prices(tickers=selected_tickers)
                all_labels = calculate_labels_for_all_tickers(price_history)
                if all_labels.empty:
                    filtered_labels = all_labels
                else:
                    filtered_labels = all_labels[
                        (all_labels["date"].dt.date >= start_date)
                        & (all_labels["date"].dt.date <= end_date)
                    ].reset_index(drop=True)
                rows_written = upsert_labels(filtered_labels)

            st.success(
                f"Processed {len(selected_tickers):,} tickers and created/updated "
                f"{rows_written:,} label rows for {start_date} through {end_date}."
            )

    st.subheader("Stored label coverage")
    label_summary = get_label_summary()
    if label_summary.empty:
        st.info("No stored labels yet.")
    else:
        st.dataframe(label_summary, width="stretch", hide_index=True)

    st.subheader("Label preview")
    labels = load_labels(tickers=selected_tickers or None, start_date=start_date, end_date=end_date)
    if labels.empty:
        st.info("No stored label rows match the current filters.")
        return

    st.dataframe(labels.tail(500), width="stretch", hide_index=True)

    st.subheader("Missing values by label")
    label_value_columns = [column for column in labels.columns if column not in {"date", "ticker", "created_at", "updated_at"}]
    missing_counts = (
        labels[label_value_columns]
        .isna()
        .sum()
        .rename("missing_values")
        .reset_index()
        .rename(columns={"index": "label"})
    )
    missing_counts = missing_counts[missing_counts["missing_values"] > 0].sort_values(
        ["missing_values", "label"], ascending=[False, True]
    )
    if missing_counts.empty:
        st.success("No missing values in the current label view.")
    else:
        st.dataframe(missing_counts, width="stretch", hide_index=True)

    st.download_button(
        "Export labels to CSV",
        data=labels.to_csv(index=False).encode("utf-8"),
        file_name="gorpiq_labels.csv",
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
        _render_label_generation()
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
