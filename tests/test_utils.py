from gorpiq.utils import combine_ticker_sources, normalize_tickers, parse_tickers_from_text


def test_normalize_tickers_deduplicates_and_uppercases() -> None:
    assert normalize_tickers("aapl, msft\nAAPL; spy") == ["AAPL", "MSFT", "SPY"]


def test_parse_tickers_from_text_accepts_commas_spaces_tabs_and_newlines() -> None:
    text = "aapl, MSFT\tNVDA\nspy  qqq,,"
    assert parse_tickers_from_text(text) == ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]


def test_parse_tickers_from_text_removes_blanks_and_preserves_first_seen_order() -> None:
    text = "  msft\n\n aapl MSFT, nvda aapl "
    assert parse_tickers_from_text(text) == ["MSFT", "AAPL", "NVDA"]


def test_combine_ticker_sources_deduplicates_manual_and_csv_sources() -> None:
    manual = "aapl msft\nnvda"
    csv = ["MSFT", "tsla, aapl", "qqq"]
    assert combine_ticker_sources(manual, csv) == ["AAPL", "MSFT", "NVDA", "TSLA", "QQQ"]


def test_combine_ticker_sources_handles_empty_inputs() -> None:
    assert combine_ticker_sources("", []) == []
