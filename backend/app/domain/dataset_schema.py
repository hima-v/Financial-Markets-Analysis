from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalColumns:
    date: str = "date"
    symbol: str = "symbol"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"
    vwap: str = "vwap"


CANON = CanonicalColumns()

REQUIRED_COLUMNS: tuple[str, ...] = (CANON.date, CANON.symbol, CANON.close)
RECOMMENDED_COLUMNS: tuple[str, ...] = (CANON.open, CANON.high, CANON.low, CANON.volume)

COLUMN_ALIASES: dict[str, str] = {
    "DATE": CANON.date,
    "Date": CANON.date,
    "date": CANON.date,
    "SYMBOL": CANON.symbol,
    "Symbol": CANON.symbol,
    "symbol": CANON.symbol,
    "OPEN": CANON.open,
    "Open": CANON.open,
    "open": CANON.open,
    "HIGH": CANON.high,
    "High": CANON.high,
    "high": CANON.high,
    "LOW": CANON.low,
    "Low": CANON.low,
    "low": CANON.low,
    "CLOSE": CANON.close,
    "Close": CANON.close,
    "close": CANON.close,
    "VOLUME": CANON.volume,
    "Volume": CANON.volume,
    "volume": CANON.volume,
    "VWAP": CANON.vwap,
    "vwap": CANON.vwap,
}


def normalize_column_name(name: str) -> str:
    trimmed = str(name).strip()
    return COLUMN_ALIASES.get(trimmed, trimmed)

