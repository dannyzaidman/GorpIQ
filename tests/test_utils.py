from gorpiq.utils import normalize_tickers


def test_normalize_tickers_deduplicates_and_uppercases() -> None:
    assert normalize_tickers("aapl, msft\nAAPL; spy") == ["AAPL", "MSFT", "SPY"]
