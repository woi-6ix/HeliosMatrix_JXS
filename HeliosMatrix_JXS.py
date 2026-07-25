















"""
HELIOS Matrix Options
Trading dashboard for regime classification + option spread scanning.
"""

from __future__ import annotations

import math
import html
import os

# Streamlit Cloud stability for transformer/PyTorch models.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import uuid
import logging
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import feedparser
except Exception:
    feedparser = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

# Extra workaround for Streamlit + torch.classes watcher bug on cloud deploys.
try:
    import torch
    torch.classes.__path__ = []
except Exception:
    pass


# =============================================================================
# APP CONFIG
# =============================================================================

st.set_page_config(
    page_title="HELIOS Options Matrix",
    page_icon="ð",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.getLogger("yfinance").setLevel(logging.WARNING)

JOURNAL_PATH = "paper_trade_journal.csv"
RISK_FREE_RATE = 0.045  # Simple default approximation for Black-Scholes Greeks.
CONTRACT_MULTIPLIER = 100

JOURNAL_COLUMNS = [
    "trade_id",
    "entry_date",
    "ticker",
    "strategy",
    "expiration",
    "dte",
    "regime",
    "direction_bias",
    "strikes",
    "model_score",
    "rule_score",
    "reason",
    "entry_type",
    "entry_premium",
    "max_profit",
    "max_loss",
    "net_delta",
    "net_theta",
    "net_vega",
    "iv_rank_proxy",
    "rv20",
    "atr_pct",
    "exit_date",
    "exit_premium",
    "realized_pnl",
    "exit_reason",
    "label",
    "mistake_notes",
    "paper_only",
]

MODEL_FEATURES = [
    "dte",
    "width",
    "premium",
    "max_profit",
    "max_loss",
    "reward_to_risk",
    "net_delta",
    "net_theta",
    "net_vega",
    "short_delta_abs",
    "short_moneyness_pct",
    "liquidity_score",
    "avg_bid_ask_pct",
    "iv_rank_proxy",
    "atm_iv",
    "rv20",
    "rv60",
    "atr_pct",
    "trend_strength",
    "regime_code",
    "strategy_code",
]

STRATEGY_CODE = {
    "Bull Put Credit Spread": 1,
    "Bear Call Credit Spread": 2,
    "Bull Call Debit Spread": 3,
    "Bear Put Debit Spread": 4,
    "Iron Condor Candidate": 5,
}

REGIME_CODE = {
    "Sideways / High IV": 1,
    "Sideways / Low IV": 2,
    "Bullish / High IV": 3,
    "Bullish / Low IV": 4,
    "Bullish / Normal IV": 5,
    "Bearish / High IV": 6,
    "Bearish / Low IV": 7,
    "Bearish / Normal IV": 8,
    "Volatile / Directional": 9,
    "Event Risk": 10,
    "Mixed / Wait": 11,
}


# =============================================================================
# STYLE
# =============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #B266FF;
        }
        .stButton>button, .stDownloadButton>button {
            background-color: #800080;
            color: #FFFFFF;
            border-radius: 7px;
            border: 1px solid #B266FF;
            font-weight: 600;
        }
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div,
        .stMultiSelect>div>div,
        .stTextArea>div>textarea {
            background-color: #111111;
            color: #FFFFFF;
            border: 1px solid #800080;
        }
        [data-testid="stMetricValue"] {
            color: #FFFFFF;
        }
        [data-testid="stMetricLabel"] {
            color: #B266FF;
        }
        .jxs-card {
            border: 1px solid #3B0A45;
            border-radius: 12px;
            padding: 18px;
            background: linear-gradient(135deg, #0B0B0B 0%, #1A061F 100%);
            margin-bottom: 12px;
        }
        .small-note {
            font-size: 0.90rem;
            color: #C7C7C7;
        }
        .news-readout-metrics {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.15rem 0 0.9rem 0;
        }
        .news-mini-metric {
            border: 1px solid #3B0A45;
            border-radius: 10px;
            padding: 0.65rem 0.75rem;
            background: linear-gradient(135deg, #0B0B0B 0%, #1A061F 100%);
            min-width: 0;
            overflow-wrap: anywhere;
        }
        .news-mini-label {
            color: #B266FF;
            font-size: 0.78rem;
            line-height: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            white-space: normal;
        }
        .news-mini-value {
            color: #FFFFFF;
            font-size: 1.05rem;
            line-height: 1.25rem;
            font-weight: 700;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        @media (max-width: 900px) {
            .news-readout-metrics {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# BASIC HELPERS
# =============================================================================

def clean_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def safe_float(value, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def today_utc_date() -> date:
    return datetime.now(timezone.utc).date()


def dte_from_expiration(expiration: str) -> int:
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        return max((exp_date - today_utc_date()).days, 0)
    except Exception:
        return 0


def normalize_0_100(x: float, low: float, high: float) -> float:
    if not np.isfinite(x) or high == low:
        return 0.0
    return float(np.clip((x - low) / (high - low) * 100.0, 0.0, 100.0))


def format_money(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"${x:,.2f}"


# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_price_data(ticker: str, period: str = "2y") -> Tuple[pd.DataFrame, Optional[str]]:
    ticker = clean_ticker(ticker)
    if not ticker:
        return pd.DataFrame(), "Ticker cannot be blank."
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        df = flatten_yfinance_columns(df)
        if df.empty:
            return pd.DataFrame(), f"No price data returned for {ticker}. Try another Yahoo Finance symbol."
        if "Close" not in df.columns:
            return pd.DataFrame(), f"No Close column found for {ticker}."
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna(subset=["Close"])
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Price download error for {ticker}: {e}"


@st.cache_data(ttl=1800, show_spinner=False)
def get_option_expirations(ticker: str) -> Tuple[List[str], Optional[str]]:
    ticker = clean_ticker(ticker)
    try:
        tk = yf.Ticker(ticker)
        expirations = list(tk.options or [])
        if not expirations:
            return [], f"No option expirations returned for {ticker}. Yahoo Finance may not support this symbol's option chain."
        return expirations, None
    except Exception as e:
        return [], f"Option expiration error for {ticker}: {e}"


@st.cache_data(ttl=900, show_spinner=False)
def get_option_chain(ticker: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    ticker = clean_ticker(ticker)
    try:
        chain = yf.Ticker(ticker).option_chain(expiration)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        if calls.empty and puts.empty:
            return pd.DataFrame(), pd.DataFrame(), f"Empty option chain for {ticker} {expiration}."
        return calls, puts, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"Option chain error for {ticker} {expiration}: {e}"


@st.cache_data(ttl=3600, show_spinner=False)
def get_calendar_event_date(ticker: str) -> Optional[date]:
    """Best-effort earnings/event date. Often unavailable for ETFs/indexes."""
    try:
        cal = yf.Ticker(clean_ticker(ticker)).calendar
        if cal is None:
            return None

        # yfinance can return dict-like or DataFrame-like objects depending on version/data.
        if isinstance(cal, dict):
            for key in ["Earnings Date", "Earnings Average", "Earnings Low", "Earnings High"]:
                val = cal.get(key)
                if val is not None:
                    if isinstance(val, (list, tuple)) and val:
                        val = val[0]
                    parsed = pd.to_datetime(val, errors="coerce")
                    if pd.notna(parsed):
                        return parsed.date()

        if isinstance(cal, pd.DataFrame) and not cal.empty:
            # Common format: index has event names and first column has values.
            for idx in cal.index:
                if "earn" in str(idx).lower():
                    val = cal.loc[idx].dropna().iloc[0]
                    parsed = pd.to_datetime(val, errors="coerce")
                    if pd.notna(parsed):
                        return parsed.date()
    except Exception:
        return None
    return None


# =============================================================================
# TECHNICAL INDICATORS + REGIME CLASSIFIER
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Daily Return"] = out["Close"].pct_change()
    out["Daily Return %"] = out["Daily Return"] * 100

    for n in [5, 10, 20, 50, 100, 200]:
        out[f"SMA_{n}"] = out["Close"].rolling(n, min_periods=max(2, n // 2)).mean()

    high_low = out["High"] - out["Low"]
    high_prev_close = (out["High"] - out["Close"].shift(1)).abs()
    low_prev_close = (out["Low"] - out["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    out["TR"] = true_range
    out["ATR_14"] = out["TR"].rolling(14, min_periods=5).mean()
    out["ATR_%"] = out["ATR_14"] / out["Close"] * 100

    for n in [10, 20, 30, 60, 120, 252]:
        out[f"RV_{n}"] = out["Daily Return"].rolling(n, min_periods=max(5, n // 3)).std() * np.sqrt(252)
        out[f"RV_{n}_%"] = out[f"RV_{n}"] * 100

    out["SMA_20_slope"] = out["SMA_20"].pct_change(10) * 100
    out["SMA_50_slope"] = out["SMA_50"].pct_change(20) * 100
    out["Trend_20d_%"] = out["Close"].pct_change(20) * 100
    out["Trend_50d_%"] = out["Close"].pct_change(50) * 100

    # A simple normalized trend score: direction + distance from moving average.
    out["Trend_Strength"] = (
        (out["Close"] - out["SMA_50"]) / out["ATR_14"].replace(0, np.nan)
    ).clip(-10, 10)

    return out


def atm_iv_from_chain(calls: pd.DataFrame, puts: pd.DataFrame, underlying_price: float) -> float:
    pieces = []
    for raw in [calls, puts]:
        if raw is None or raw.empty or "impliedVolatility" not in raw.columns:
            continue
        df = raw.copy()
        df["dist"] = (df["strike"] - underlying_price).abs()
        df = df.sort_values("dist").head(4)
        pieces.append(df["impliedVolatility"])
    if not pieces:
        return np.nan
    iv = pd.concat(pieces, ignore_index=True)
    iv = pd.to_numeric(iv, errors="coerce")
    iv = iv[(iv > 0) & (iv < 5)]
    return float(iv.median()) if not iv.empty else np.nan


def iv_rank_proxy(current_atm_iv: float, indicators: pd.DataFrame) -> float:
    """
    Proxy for IV rank because free Yahoo chains provide current implied vol,
    not a clean historical IV time series. This compares current ATM IV to the
    historical realized-volatility range.
    """
    if not np.isfinite(current_atm_iv):
        return np.nan
    rv_range = indicators["RV_20"].dropna().tail(252)
    if len(rv_range) < 30:
        return np.nan
    lo, hi = float(rv_range.min()), float(rv_range.max())
    if hi <= lo:
        return np.nan
    return float(np.clip((current_atm_iv - lo) / (hi - lo) * 100.0, 0, 100))


def classify_regime(
    indicators: pd.DataFrame,
    atm_iv: float,
    iv_rank: float,
    earnings_date: Optional[date],
    event_risk_days: int,
) -> Dict[str, object]:
    latest = indicators.dropna(subset=["Close"]).iloc[-1]
    close = safe_float(latest.get("Close"))
    sma20 = safe_float(latest.get("SMA_20"))
    sma50 = safe_float(latest.get("SMA_50"))
    sma200 = safe_float(latest.get("SMA_200"))
    trend_20 = safe_float(latest.get("Trend_20d_%"), 0)
    trend_strength = safe_float(latest.get("Trend_Strength"), 0)
    atr_pct = safe_float(latest.get("ATR_%"), np.nan)
    rv20 = safe_float(latest.get("RV_20_%"), np.nan)
    rv60 = safe_float(latest.get("RV_60_%"), np.nan)

    event_risk = False
    days_to_event = None
    if earnings_date is not None:
        days_to_event = (earnings_date - today_utc_date()).days
        event_risk = 0 <= days_to_event <= event_risk_days

    iv_bucket = "Unknown IV"
    if np.isfinite(iv_rank):
        if iv_rank >= 65:
            iv_bucket = "High IV"
        elif iv_rank <= 35:
            iv_bucket = "Low IV"
        else:
            iv_bucket = "Normal IV"

    bullish = close > sma20 > sma50 and trend_20 > 1.0
    bearish = close < sma20 < sma50 and trend_20 < -1.0
    sideways = abs(trend_strength) < 1.25 and abs(trend_20) < max(1.25, (atr_pct if np.isfinite(atr_pct) else 1.0) * 1.25)
    volatile = np.isfinite(rv20) and np.isfinite(rv60) and rv20 > rv60 * 1.25

    if event_risk:
        regime = "Event Risk"
        setup = "Avoid or reduce paper size"
        reason = f"Possible event within {days_to_event} days; IV and gap risk can distort spread pricing."
        direction_bias = "Neutral / Cautious"
    elif sideways and iv_bucket == "High IV":
        regime = "Sideways / High IV"
        setup = "Iron condor or credit-spread candidate"
        reason = "Trend is weak while IV proxy is elevated; premium-selling structures may be worth scanning."
        direction_bias = "Neutral"
    elif sideways and iv_bucket == "Low IV":
        regime = "Sideways / Low IV"
        setup = "Butterfly or small debit-spread candidate"
        reason = "Trend is weak and IV proxy is low; defined-risk debit structures may be cleaner than selling premium."
        direction_bias = "Neutral"
    elif bullish:
        regime = f"Bullish / {iv_bucket.replace('Unknown IV', 'Normal IV')}"
        setup = "Call debit spread or bull put credit-spread candidate"
        reason = "Price is above key moving averages with positive recent trend."
        direction_bias = "Bullish"
    elif bearish:
        regime = f"Bearish / {iv_bucket.replace('Unknown IV', 'Normal IV')}"
        setup = "Put debit spread or bear call credit-spread candidate"
        reason = "Price is below key moving averages with negative recent trend."
        direction_bias = "Bearish"
    elif volatile:
        regime = "Volatile / Directional"
        setup = "Reduce size or wait for cleaner setup"
        reason = "Short-term realized volatility is running above medium-term volatility."
        direction_bias = "Unclear"
    else:
        regime = "Mixed / Wait"
        setup = "No clean edge; paper-watchlist only"
        reason = "Trend, volatility, and IV proxy are not aligned enough for a clean setup."
        direction_bias = "Mixed"

    return {
        "ticker_price": close,
        "regime": regime,
        "suggested_setup": setup,
        "reason": reason,
        "direction_bias": direction_bias,
        "atm_iv": atm_iv,
        "iv_rank_proxy": iv_rank,
        "rv20": rv20,
        "rv60": rv60,
        "atr_pct": atr_pct,
        "trend_strength": trend_strength,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "earnings_date": earnings_date,
        "days_to_event": days_to_event,
    }


# =============================================================================
# BLACK-SCHOLES GREEKS
# =============================================================================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str,
    r: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """Approximate per-share Greeks for a long option position."""
    try:
        S = float(S)
        K = float(K)
        T = max(float(T), 1 / 365)
        sigma = float(sigma)
        if S <= 0 or K <= 0 or sigma <= 0 or sigma > 5:
            return {"delta": np.nan, "theta": np.nan, "vega": np.nan}

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type.lower() == "call":
            delta = norm_cdf(d1)
            theta = (
                -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
                - r * K * math.exp(-r * T) * norm_cdf(d2)
            ) / 365.0
        else:
            delta = norm_cdf(d1) - 1
            theta = (
                -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
                + r * K * math.exp(-r * T) * norm_cdf(-d2)
            ) / 365.0

        vega = S * norm_pdf(d1) * math.sqrt(T) / 100.0
        return {"delta": delta, "theta": theta, "vega": vega}
    except Exception:
        return {"delta": np.nan, "theta": np.nan, "vega": np.nan}


def clean_options_dataframe(
    raw: pd.DataFrame,
    option_type: str,
    expiration: str,
    underlying_price: float,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    needed = ["strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["option_type"] = option_type.lower()
    df["expiration"] = expiration
    df["dte"] = dte_from_expiration(expiration)
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["bid_ask_spread"] = df["ask"] - df["bid"]
    df["bid_ask_pct"] = np.where(df["mid"] > 0, df["bid_ask_spread"] / df["mid"] * 100, np.nan)
    df["moneyness_pct"] = (df["strike"] - underlying_price) / underlying_price * 100

    T = max(df["dte"].iloc[0] / 365.0, 1 / 365)
    greek_rows = []
    for _, row in df.iterrows():
        greek_rows.append(
            black_scholes_greeks(
                S=underlying_price,
                K=safe_float(row["strike"]),
                T=T,
                sigma=safe_float(row["impliedVolatility"]),
                option_type=option_type,
            )
        )
    greeks = pd.DataFrame(greek_rows, index=df.index)
    df["delta"] = greeks["delta"]
    df["theta"] = greeks["theta"]
    df["vega"] = greeks["vega"]

    # Conservative liquidity score: tight spreads + open interest + valid quote.
    spread_score = 100 - df["bid_ask_pct"].clip(lower=0, upper=100).fillna(100)
    oi_score = np.log1p(df["openInterest"].fillna(0)).clip(0, 8) / 8 * 100
    vol_score = np.log1p(df["volume"].fillna(0)).clip(0, 8) / 8 * 100
    valid_quote = ((df["bid"] > 0) & (df["ask"] > df["bid"])).astype(float) * 100
    df["liquidity_score"] = (0.50 * spread_score + 0.25 * oi_score + 0.15 * vol_score + 0.10 * valid_quote).round(2)

    return df.sort_values("strike").reset_index(drop=True)


# =============================================================================
# SPREAD GENERATION
# =============================================================================

@dataclass
class SpreadLeg:
    row: pd.Series
    qty: int  # +1 long, -1 short


def leg_symbol(row: pd.Series) -> str:
    side = "C" if row.get("option_type") == "call" else "P"
    return f"{row.get('strike'):.2f}{side}"


def combine_greeks(legs: List[SpreadLeg]) -> Dict[str, float]:
    out = {}
    for g in ["delta", "theta", "vega"]:
        val = 0.0
        for leg in legs:
            leg_val = safe_float(leg.row.get(g), 0)
            val += leg.qty * leg_val
        out[f"net_{g}"] = val
    return out


def average_liquidity(legs: List[SpreadLeg]) -> float:
    vals = [safe_float(leg.row.get("liquidity_score"), 0) for leg in legs]
    return float(np.mean(vals)) if vals else 0.0


def average_bid_ask_pct(legs: List[SpreadLeg]) -> float:
    vals = [safe_float(leg.row.get("bid_ask_pct"), np.nan) for leg in legs]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def spread_common_fields(
    ticker: str,
    strategy: str,
    expiration: str,
    entry_type: str,
    premium: float,
    width: float,
    max_profit: float,
    max_loss: float,
    legs: List[SpreadLeg],
    underlying_price: float,
    regime_info: Dict[str, object],
) -> Dict[str, object]:
    greeks = combine_greeks(legs)
    short_legs = [leg for leg in legs if leg.qty < 0]
    short_delta_abs = abs(safe_float(short_legs[0].row.get("delta"), np.nan)) if short_legs else np.nan
    short_strike = safe_float(short_legs[0].row.get("strike"), np.nan) if short_legs else np.nan
    short_moneyness_pct = (short_strike - underlying_price) / underlying_price * 100 if np.isfinite(short_strike) else np.nan

    strikes = " / ".join([("Long " if leg.qty > 0 else "Short ") + leg_symbol(leg.row) for leg in legs])
    reward_to_risk = max_profit / max_loss if max_loss and max_loss > 0 else np.nan

    return {
        "ticker": ticker,
        "strategy": strategy,
        "expiration": expiration,
        "dte": dte_from_expiration(expiration),
        "entry_type": entry_type,
        "premium": premium,
        "width": width,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "max_profit_$": max_profit * CONTRACT_MULTIPLIER,
        "max_loss_$": max_loss * CONTRACT_MULTIPLIER,
        "reward_to_risk": reward_to_risk,
        "underlying_price": underlying_price,
        "strikes": strikes,
        "short_delta_abs": short_delta_abs,
        "short_moneyness_pct": short_moneyness_pct,
        "liquidity_score": average_liquidity(legs),
        "avg_bid_ask_pct": average_bid_ask_pct(legs),
        "net_delta": greeks["net_delta"],
        "net_theta": greeks["net_theta"],
        "net_vega": greeks["net_vega"],
        "regime": regime_info.get("regime"),
        "direction_bias": regime_info.get("direction_bias"),
        "iv_rank_proxy": regime_info.get("iv_rank_proxy"),
        "atm_iv": regime_info.get("atm_iv"),
        "rv20": regime_info.get("rv20"),
        "rv60": regime_info.get("rv60"),
        "atr_pct": regime_info.get("atr_pct"),
        "trend_strength": regime_info.get("trend_strength"),
        "regime_code": REGIME_CODE.get(str(regime_info.get("regime")), 0),
        "strategy_code": STRATEGY_CODE.get(strategy, 0),
    }


def generate_verticals_for_expiration(
    ticker: str,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    expiration: str,
    underlying_price: float,
    regime_info: Dict[str, object],
    min_width: float,
    max_width: float,
    max_bid_ask_pct: float,
    min_open_interest: int,
    min_credit_or_debit: float,
) -> pd.DataFrame:
    spreads: List[Dict[str, object]] = []

    calls = calls.copy()
    puts = puts.copy()

    liquid_calls = calls[
        (calls["bid"] > 0)
        & (calls["ask"] > calls["bid"])
        & (calls["openInterest"].fillna(0) >= min_open_interest)
        & (calls["bid_ask_pct"].fillna(999) <= max_bid_ask_pct)
    ].sort_values("strike")

    liquid_puts = puts[
        (puts["bid"] > 0)
        & (puts["ask"] > puts["bid"])
        & (puts["openInterest"].fillna(0) >= min_open_interest)
        & (puts["bid_ask_pct"].fillna(999) <= max_bid_ask_pct)
    ].sort_values("strike")

    # CALL verticals: lower strike + higher strike.
    for i, lower in liquid_calls.iterrows():
        for j, higher in liquid_calls[liquid_calls["strike"] > lower["strike"]].iterrows():
            width = safe_float(higher["strike"] - lower["strike"])
            if width < min_width or width > max_width:
                continue

            # Bull Call Debit Spread: long lower call, short higher call.
            debit = safe_float(lower["ask"]) - safe_float(higher["bid"])
            if debit >= min_credit_or_debit and debit > 0 and debit < width:
                max_profit = width - debit
                max_loss = debit
                legs = [SpreadLeg(lower, +1), SpreadLeg(higher, -1)]
                spreads.append(
                    spread_common_fields(
                        ticker, "Bull Call Debit Spread", expiration, "debit", debit, width,
                        max_profit, max_loss, legs, underlying_price, regime_info
                    )
                )

            # Bear Call Credit Spread: short lower call, long higher call.
            credit = safe_float(lower["bid"]) - safe_float(higher["ask"])
            if credit >= min_credit_or_debit and credit > 0 and credit < width:
                max_profit = credit
                max_loss = width - credit
                legs = [SpreadLeg(lower, -1), SpreadLeg(higher, +1)]
                spreads.append(
                    spread_common_fields(
                        ticker, "Bear Call Credit Spread", expiration, "credit", credit, width,
                        max_profit, max_loss, legs, underlying_price, regime_info
                    )
                )

    # PUT verticals: lower strike + higher strike.
    for i, lower in liquid_puts.iterrows():
        for j, higher in liquid_puts[liquid_puts["strike"] > lower["strike"]].iterrows():
            width = safe_float(higher["strike"] - lower["strike"])
            if width < min_width or width > max_width:
                continue

            # Bear Put Debit Spread: long higher put, short lower put.
            debit = safe_float(higher["ask"]) - safe_float(lower["bid"])
            if debit >= min_credit_or_debit and debit > 0 and debit < width:
                max_profit = width - debit
                max_loss = debit
                legs = [SpreadLeg(higher, +1), SpreadLeg(lower, -1)]
                spreads.append(
                    spread_common_fields(
                        ticker, "Bear Put Debit Spread", expiration, "debit", debit, width,
                        max_profit, max_loss, legs, underlying_price, regime_info
                    )
                )

            # Bull Put Credit Spread: short higher put, long lower put.
            credit = safe_float(higher["bid"]) - safe_float(lower["ask"])
            if credit >= min_credit_or_debit and credit > 0 and credit < width:
                max_profit = credit
                max_loss = width - credit
                legs = [SpreadLeg(higher, -1), SpreadLeg(lower, +1)]
                spreads.append(
                    spread_common_fields(
                        ticker, "Bull Put Credit Spread", expiration, "credit", credit, width,
                        max_profit, max_loss, legs, underlying_price, regime_info
                    )
                )

    return pd.DataFrame(spreads)


def generate_iron_condor_candidates(verticals: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
    if verticals.empty:
        return pd.DataFrame()

    puts = verticals[verticals["strategy"] == "Bull Put Credit Spread"].copy()
    calls = verticals[verticals["strategy"] == "Bear Call Credit Spread"].copy()
    if puts.empty or calls.empty:
        return pd.DataFrame()

    rows = []
    grouped_puts = puts.groupby(["ticker", "expiration", "width"])
    for (ticker, expiration, width), pgroup in grouped_puts:
        cgroup = calls[
            (calls["ticker"] == ticker)
            & (calls["expiration"] == expiration)
            & (calls["width"] == width)
        ]
        if cgroup.empty:
            continue

        # Keep a manageable number of pairings.
        ptop = pgroup.sort_values(["rule_score" if "rule_score" in pgroup.columns else "liquidity_score"], ascending=False).head(10)
        ctop = cgroup.sort_values(["rule_score" if "rule_score" in cgroup.columns else "liquidity_score"], ascending=False).head(10)

        for _, p in ptop.iterrows():
            for _, c in ctop.iterrows():
                credit = safe_float(p["premium"]) + safe_float(c["premium"])
                max_profit = credit
                max_loss = width - credit
                if credit <= 0 or max_loss <= 0:
                    continue
                rows.append({
                    "ticker": ticker,
                    "strategy": "Iron Condor Candidate",
                    "expiration": expiration,
                    "dte": p["dte"],
                    "entry_type": "credit",
                    "premium": credit,
                    "width": width,
                    "max_profit": max_profit,
                    "max_loss": max_loss,
                    "max_profit_$": max_profit * CONTRACT_MULTIPLIER,
                    "max_loss_$": max_loss * CONTRACT_MULTIPLIER,
                    "reward_to_risk": max_profit / max_loss,
                    "underlying_price": p["underlying_price"],
                    "strikes": f"PUT SIDE: {p['strikes']}  ||  CALL SIDE: {c['strikes']}",
                    "short_delta_abs": np.mean([p["short_delta_abs"], c["short_delta_abs"]]),
                    "short_moneyness_pct": np.nan,
                    "liquidity_score": np.mean([p["liquidity_score"], c["liquidity_score"]]),
                    "avg_bid_ask_pct": np.mean([p["avg_bid_ask_pct"], c["avg_bid_ask_pct"]]),
                    "net_delta": p["net_delta"] + c["net_delta"],
                    "net_theta": p["net_theta"] + c["net_theta"],
                    "net_vega": p["net_vega"] + c["net_vega"],
                    "regime": p["regime"],
                    "direction_bias": "Neutral",
                    "iv_rank_proxy": p["iv_rank_proxy"],
                    "atm_iv": p["atm_iv"],
                    "rv20": p["rv20"],
                    "rv60": p["rv60"],
                    "atr_pct": p["atr_pct"],
                    "trend_strength": p["trend_strength"],
                    "regime_code": p["regime_code"],
                    "strategy_code": STRATEGY_CODE["Iron Condor Candidate"],
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["liquidity_score", "reward_to_risk"], ascending=False).head(max_rows)


def score_spreads_rule_based(spreads: pd.DataFrame, regime: Optional[str] = None) -> pd.DataFrame:
    if spreads.empty:
        return spreads

    out = spreads.copy()
    scores = []
    reasons = []
    for _, row in out.iterrows():
        strategy = row["strategy"]
        active_regime = regime or str(row.get("regime", ""))
        score = 45.0
        reason_bits = []

        # Regime fit.
        if active_regime == "Sideways / High IV" and strategy in ["Iron Condor Candidate", "Bull Put Credit Spread", "Bear Call Credit Spread"]:
            score += 24
            reason_bits.append("regime supports premium-selling")
        elif active_regime == "Sideways / Low IV" and strategy in ["Bull Call Debit Spread", "Bear Put Debit Spread"]:
            score += 10
            reason_bits.append("low IV favors defined debit structures")
        elif active_regime.startswith("Bullish") and strategy in ["Bull Call Debit Spread", "Bull Put Credit Spread"]:
            score += 22
            reason_bits.append("bullish regime fit")
        elif active_regime.startswith("Bearish") and strategy in ["Bear Put Debit Spread", "Bear Call Credit Spread"]:
            score += 22
            reason_bits.append("bearish regime fit")
        elif active_regime in ["Event Risk", "Mixed / Wait", "Volatile / Directional"]:
            score -= 12
            reason_bits.append("cautious regime")
        else:
            score -= 5
            reason_bits.append("weaker regime fit")

        # Liquidity.
        liq = safe_float(row.get("liquidity_score"), 0)
        score += np.clip((liq - 50) / 2.5, -10, 15)
        if liq >= 75:
            reason_bits.append("good liquidity")
        elif liq < 50:
            reason_bits.append("weak liquidity")

        # Credit/debit economics.
        rr = safe_float(row.get("reward_to_risk"), np.nan)
        if np.isfinite(rr):
            score += np.clip(rr * 12, -5, 16)
            if rr >= 0.35:
                reason_bits.append("reasonable reward/risk")

        # Short delta preference for credit spreads / IC.
        short_delta = safe_float(row.get("short_delta_abs"), np.nan)
        if row.get("entry_type") == "credit" and np.isfinite(short_delta):
            if 0.10 <= short_delta <= 0.35:
                score += 10
                reason_bits.append(f"short delta {short_delta:.2f}")
            elif short_delta > 0.45:
                score -= 12
                reason_bits.append(f"aggressive short delta {short_delta:.2f}")

        # Theta/vega behavior.
        if row.get("entry_type") == "credit":
            if safe_float(row.get("net_theta"), 0) > 0:
                score += 5
                reason_bits.append("positive theta")
            if safe_float(row.get("net_vega"), 0) < 0:
                score += 4
                reason_bits.append("short vega")
        else:
            if abs(safe_float(row.get("net_delta"), 0)) > 0.10:
                score += 4
                reason_bits.append("directional delta")

        # DTE preference.
        dte = safe_float(row.get("dte"), np.nan)
        if np.isfinite(dte):
            if dte == 0 and strategy == "Iron Condor Candidate":
                # 0DTE iron condors are a primary HELIOS workflow, so do not apply the generic short-DTE penalty.
                score += 6
                reason_bits.append("0DTE IC focus")
                if abs(safe_float(row.get("net_delta"), 0)) <= 0.12:
                    score += 4
                    reason_bits.append("balanced 0DTE delta")
            elif dte == 0:
                score -= 4
                reason_bits.append("0DTE gamma risk")
            elif 20 <= dte <= 60:
                score += 7
                reason_bits.append("DTE in preferred range")
            elif dte < 7:
                score -= 6
                reason_bits.append("short-DTE gamma risk")

        scores.append(float(np.clip(score, 0, 100)))
        reasons.append(", ".join(reason_bits[:5]))

    out["rule_score"] = scores
    out["reason"] = reasons
    return out


# =============================================================================
# JOURNAL + LABELING + ML RANKER
# =============================================================================

def empty_journal() -> pd.DataFrame:
    return pd.DataFrame(columns=JOURNAL_COLUMNS)


def load_journal(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            for col in JOURNAL_COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            return df[JOURNAL_COLUMNS]
        except Exception as e:
            st.error(f"Could not read uploaded journal: {e}")
            return empty_journal()

    if os.path.exists(JOURNAL_PATH):
        try:
            df = pd.read_csv(JOURNAL_PATH)
            for col in JOURNAL_COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            return df[JOURNAL_COLUMNS]
        except Exception:
            return empty_journal()
    return empty_journal()


def save_journal(df: pd.DataFrame) -> None:
    out = df.copy()
    for col in JOURNAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out[JOURNAL_COLUMNS].to_csv(JOURNAL_PATH, index=False)


def label_trade(row: pd.Series) -> str:
    manual = str(row.get("label", "")).strip()
    if manual and manual.lower() not in ["nan", "none", "unknown"]:
        return manual

    entry_type = str(row.get("entry_type", "")).lower().strip()
    entry_premium = safe_float(row.get("entry_premium"), np.nan)
    exit_premium = safe_float(row.get("exit_premium"), np.nan)
    max_profit = safe_float(row.get("max_profit"), np.nan)
    max_loss = safe_float(row.get("max_loss"), np.nan)
    realized_pnl = safe_float(row.get("realized_pnl"), np.nan)
    exit_reason = str(row.get("exit_reason", "")).lower()

    # Try to infer P/L if user entered entry/exit premium only.
    if not np.isfinite(realized_pnl) and np.isfinite(entry_premium) and np.isfinite(exit_premium):
        if entry_type == "credit":
            realized_pnl = entry_premium - exit_premium
        elif entry_type == "debit":
            realized_pnl = exit_premium - entry_premium

    if "max loss" in exit_reason or "assignment" in exit_reason:
        return "hit_max_loss"

    if np.isfinite(realized_pnl) and np.isfinite(max_loss) and max_loss > 0 and realized_pnl <= -0.95 * max_loss:
        return "hit_max_loss"

    if entry_type == "credit" and np.isfinite(entry_premium) and np.isfinite(exit_premium):
        if exit_premium <= 0.50 * entry_premium:
            return "hit_50_profit"

    if np.isfinite(realized_pnl) and np.isfinite(max_profit) and max_profit > 0:
        if realized_pnl >= 0.50 * max_profit:
            return "hit_50_profit"
        if realized_pnl > 0:
            # If exit reason says expiry, make that explicit.
            if "expir" in exit_reason:
                return "expired_profitable"
            return "closed_profitable"
        return "closed_loss"

    return "unknown"


def add_labels_to_journal(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["label"] = out.apply(label_trade, axis=1)
    return out


def prepare_model_training_data(journal: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], Optional[str]]:
    if journal.empty:
        return pd.DataFrame(), pd.Series(dtype=int), [], "Journal is empty. Add paper trades before training."

    df = add_labels_to_journal(journal)
    success_labels = {"hit_50_profit", "expired_profitable", "closed_profitable"}
    fail_labels = {"hit_max_loss", "closed_loss"}
    df = df[df["label"].isin(success_labels.union(fail_labels))].copy()
    if len(df) < 20:
        return pd.DataFrame(), pd.Series(dtype=int), [], "Need at least 20 labeled paper trades for a first ML ranker. More is better."

    df["target_success"] = df["label"].isin(success_labels).astype(int)
    if df["target_success"].nunique() < 2:
        return pd.DataFrame(), pd.Series(dtype=int), [], "Need both winning and losing examples to train a classifier."

    # Rebuild features from journal columns. Missing columns become NaN and are imputed.
    training = pd.DataFrame(index=df.index)
    journal_to_feature = {
        "model_score": "model_score",
        "rule_score": "rule_score",
        "entry_premium": "premium",
        "max_profit": "max_profit",
        "max_loss": "max_loss",
        "net_delta": "net_delta",
        "net_theta": "net_theta",
        "net_vega": "net_vega",
        "iv_rank_proxy": "iv_rank_proxy",
        "rv20": "rv20",
        "atr_pct": "atr_pct",
        "dte": "dte",
    }
    for src, dst in journal_to_feature.items():
        training[dst] = pd.to_numeric(df.get(src), errors="coerce")

    training["width"] = np.nan
    training["reward_to_risk"] = training["max_profit"] / training["max_loss"].replace(0, np.nan)
    training["short_delta_abs"] = np.nan
    training["short_moneyness_pct"] = np.nan
    training["liquidity_score"] = pd.to_numeric(df.get("rule_score"), errors="coerce").clip(0, 100)
    training["avg_bid_ask_pct"] = np.nan
    training["atm_iv"] = np.nan
    training["rv60"] = np.nan
    training["trend_strength"] = np.nan
    training["regime_code"] = df.get("regime", "").map(REGIME_CODE).fillna(0)
    training["strategy_code"] = df.get("strategy", "").map(STRATEGY_CODE).fillna(0)

    X = training.reindex(columns=MODEL_FEATURES)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = df["target_success"].astype(int)
    return X, y, MODEL_FEATURES, None


def train_xgboost_ranker(journal: pd.DataFrame):
    if xgb is None:
        return None, [], None, "xgboost is not installed. Add xgboost to requirements.txt."

    X, y, feature_cols, err = prepare_model_training_data(journal)
    if err:
        return None, feature_cols, None, err

    if len(X) >= 50:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = xgb.XGBClassifier(
        n_estimators=160,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=False, zero_division=0)
    metrics = {"accuracy": acc, "report": report, "rows": len(X)}
    return model, feature_cols, metrics, None


def apply_ml_ranking(spreads: pd.DataFrame, model, feature_cols: List[str]) -> pd.DataFrame:
    out = spreads.copy()
    if model is None or out.empty:
        out["model_score"] = np.nan
        out["final_score"] = out.get("rule_score", pd.Series(dtype=float))
        return out

    X = out.reindex(columns=feature_cols)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    try:
        proba = model.predict_proba(X)[:, 1] * 100.0
        out["model_score"] = proba
        out["final_score"] = 0.65 * out["model_score"] + 0.35 * out["rule_score"].fillna(50)
    except Exception:
        out["model_score"] = np.nan
        out["final_score"] = out.get("rule_score", pd.Series(dtype=float))
    return out


def journal_row_from_candidate(candidate: pd.Series, notes: str = "") -> Dict[str, object]:
    return {
        "trade_id": str(uuid.uuid4())[:8],
        "entry_date": today_utc_date().isoformat(),
        "ticker": candidate.get("ticker"),
        "strategy": candidate.get("strategy"),
        "expiration": candidate.get("expiration"),
        "dte": candidate.get("dte"),
        "regime": candidate.get("regime"),
        "direction_bias": candidate.get("direction_bias"),
        "strikes": candidate.get("strikes"),
        "model_score": candidate.get("model_score"),
        "rule_score": candidate.get("rule_score"),
        "reason": candidate.get("reason"),
        "entry_type": candidate.get("entry_type"),
        "entry_premium": candidate.get("premium"),
        "max_profit": candidate.get("max_profit"),
        "max_loss": candidate.get("max_loss"),
        "net_delta": candidate.get("net_delta"),
        "net_theta": candidate.get("net_theta"),
        "net_vega": candidate.get("net_vega"),
        "iv_rank_proxy": candidate.get("iv_rank_proxy"),
        "rv20": candidate.get("rv20"),
        "atr_pct": candidate.get("atr_pct"),
        "exit_date": "",
        "exit_premium": np.nan,
        "realized_pnl": np.nan,
        "exit_reason": "",
        "label": "unknown",
        "mistake_notes": notes,
        "paper_only": True,
    }


# =============================================================================
# OPTIONAL FINBERT SENTIMENT
# =============================================================================

NEWS_RISK_KEYWORDS = {
    "Earnings / Guidance": [
        "earnings", "eps", "revenue", "guidance", "quarterly results", "profit warning",
        "misses estimates", "beats estimates", "conference call"
    ],
    "Macro / Fed / Rates": [
        "fed", "federal reserve", "fomc", "interest rates", "rate cut", "rate hike",
        "inflation", "cpi", "ppi", "jobs report", "payrolls", "treasury yields"
    ],
    "Analyst / Rating Risk": [
        "downgrade", "upgrade", "price target", "initiates coverage", "rating cut",
        "rating raised", "analyst"
    ],
    "Legal / Regulatory": [
        "lawsuit", "sec", "doj", "ftc", "probe", "investigation", "antitrust",
        "regulatory", "settlement", "fine"
    ],
    "M&A / Corporate Action": [
        "merger", "acquisition", "takeover", "buyout", "spin off", "spinoff",
        "stock split", "dividend", "share repurchase"
    ],
    "Volatility / Shock": [
        "surges", "plunges", "crashes", "selloff", "rally", "volatile", "warning",
        "halts", "bankruptcy", "restructuring"
    ],
}


@st.cache_resource(show_spinner=False)
def get_finbert_pipeline():
    if pipeline is None:
        raise ImportError("transformers is not installed. Add transformers and torch to requirements.txt.")
    return pipeline(task="text-classification", model="ProsusAI/finbert")


def get_news_entry_text(entry) -> str:
    """Match IRIS logic: use article summary first, then title as fallback."""
    return (entry.get("summary") or entry.get("title") or "").strip()


def get_news_entry_title(entry) -> str:
    return (entry.get("title") or "Untitled").strip()


def finbert_sentiment_score_like_iris(text: str, pipe) -> Tuple[str, float, float]:
    """
    Match the IRIS scanner's displayed scoring:
    - positive = +confidence
    - negative = -confidence
    - neutral = +confidence, because IRIS only flips negative labels

    Also returns a directional score where neutral = 0.0, which is better for
    spread-bias interpretation.
    """
    if not text:
        return "neutral", 0.0, 0.0

    result = pipe(text, truncation=True)[0]
    label = str(result.get("label", "neutral")).lower()
    confidence = float(result.get("score", 0.0))

    iris_score = -confidence if label == "negative" else confidence
    directional_score = -confidence if label == "negative" else confidence if label == "positive" else 0.0
    return label, iris_score, directional_score


def detect_news_risk_flags(title: str, summary: str) -> List[str]:
    text = f"{title} {summary}".lower()
    flags = []
    for category, keywords in NEWS_RISK_KEYWORDS.items():
        if any(k in text for k in keywords):
            flags.append(category)
    return flags


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yahoo_rss_sentiment(ticker: str, keyword: str, max_articles: int = 20) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Pull Yahoo Finance RSS and score articles using the same FinBERT scoring style
    as IRIS. The previous HELIOS version produced different scores because it set
    neutral articles to 0.0 and only analyzed summary[:1800].
    """
    if feedparser is None:
        return pd.DataFrame(), "feedparser is not installed. Add feedparser to requirements.txt."

    try:
        pipe = get_finbert_pipeline()
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={clean_ticker(ticker)}&region=US&lang=en-US"
        feed = feedparser.parse(rss_url)

        if not getattr(feed, "entries", None):
            return pd.DataFrame(), None

        rows = []
        keyword_clean = str(keyword or "").strip().lower()

        for entry in feed.entries[:max_articles]:
            title = get_news_entry_title(entry)
            full_text = get_news_entry_text(entry)

            # Match IRIS keyword filtering: check both title and article text.
            if keyword_clean and keyword_clean not in full_text.lower() and keyword_clean not in title.lower():
                continue

            label, iris_score, directional_score = finbert_sentiment_score_like_iris(full_text, pipe)
            risk_flags = detect_news_risk_flags(title, full_text)

            rows.append({
                "Date": entry.get("published", "N/A"),
                "Title": title,
                "Sentiment": label,
                "Score": iris_score,
                "Directional Score": directional_score,
                "Confidence": abs(iris_score),
                "Risk Flags": ", ".join(risk_flags) if risk_flags else "None detected",
                "Link": entry.get("link", ""),
                "Full Text": full_text,
            })

        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"Sentiment error: {e}"


def build_news_spread_readout(sent_df: pd.DataFrame, regime_row: Optional[pd.Series] = None) -> Dict[str, object]:
    if sent_df is None or sent_df.empty:
        return {
            "bias": "No clear news bias",
            "volatility_warning": "No matching articles were available.",
            "spread_readout": "Do not use news as a signal until articles are available.",
            "risk_level": "Unknown",
            "warnings": [],
        }

    avg_iris_score = float(sent_df["Score"].mean())
    avg_directional_score = float(sent_df["Directional Score"].mean())
    positive_count = int((sent_df["Directional Score"] > 0).sum())
    negative_count = int((sent_df["Directional Score"] < 0).sum())
    neutral_count = int((sent_df["Directional Score"] == 0).sum())

    all_flags = []
    for val in sent_df["Risk Flags"].fillna(""):
        if val and val != "None detected":
            all_flags.extend([x.strip() for x in val.split(",") if x.strip()])

    flag_counts = pd.Series(all_flags).value_counts().to_dict() if all_flags else {}
    event_count = sum(flag_counts.values())

    # Use scanner context if available.
    iv_rank = np.nan
    regime = ""
    if regime_row is not None:
        iv_rank = safe_float(regime_row.get("IV Rank Proxy"), np.nan)
        regime = str(regime_row.get("Regime", ""))

    iv_is_high = np.isfinite(iv_rank) and iv_rank >= 65
    iv_is_low = np.isfinite(iv_rank) and iv_rank <= 35

    if event_count >= 4 or any(k in flag_counts for k in ["Earnings / Guidance", "Macro / Fed / Rates", "Volatility / Shock"]):
        risk_level = "High"
        volatility_warning = "News contains event/volatility flags. Avoid oversized short premium, and be careful with iron condors/butterflies around binary events."
    elif event_count >= 2 or abs(avg_directional_score) >= 0.45:
        risk_level = "Moderate"
        volatility_warning = "News has enough directional or event language to justify smaller paper size and wider risk checks."
    else:
        risk_level = "Low"
        volatility_warning = "No major event-risk cluster detected from the matched headlines."

    if risk_level == "High":
        bias = "Event-risk / volatility caution"
        spread_readout = "Avoid or reduce size. Prefer waiting, or use very small defined-risk paper trades only. Avoid neutral short-premium setups if earnings/macro shock risk is present."
    elif avg_directional_score >= 0.35:
        bias = "Bullish news bias"
        if iv_is_high or "High IV" in regime:
            spread_readout = "Bull put credit spread candidate: bullish news + richer IV can support defined-risk premium selling."
        elif iv_is_low or "Low IV" in regime:
            spread_readout = "Call debit spread candidate: bullish news + lower IV may make long-premium defined-risk exposure cleaner."
        else:
            spread_readout = "Bullish candidate: compare bull put credit spreads vs call debit spreads based on IV and liquidity."
    elif avg_directional_score <= -0.35:
        bias = "Bearish news bias"
        if iv_is_high or "High IV" in regime:
            spread_readout = "Bear call credit spread candidate: bearish news + richer IV can support defined-risk premium selling."
        elif iv_is_low or "Low IV" in regime:
            spread_readout = "Put debit spread candidate: bearish news + lower IV may make long-premium defined-risk exposure cleaner."
        else:
            spread_readout = "Bearish candidate: compare bear call credit spreads vs put debit spreads based on IV and liquidity."
    else:
        bias = "Neutral / mixed news bias"
        if iv_is_high or "Sideways / High IV" in regime:
            spread_readout = "Iron condor candidate only if price trend is weak and no event-risk warnings are present."
        elif iv_is_low or "Sideways / Low IV" in regime:
            spread_readout = "Butterfly candidate only if price is range-bound and liquidity is tight."
        else:
            spread_readout = "No strong news edge. Let the regime scanner and option liquidity decide; paper-watchlist is reasonable."

    warnings = [f"{name}: {count} article(s)" for name, count in flag_counts.items()]
    if negative_count > positive_count and avg_directional_score < -0.20:
        warnings.append("Negative article count is heavier than positive article count.")
    if neutral_count >= max(3, len(sent_df) // 2):
        warnings.append("Most articles are neutral, so the news signal may be weak.")

    return {
        "bias": bias,
        "volatility_warning": volatility_warning,
        "spread_readout": spread_readout,
        "risk_level": risk_level,
        "warnings": warnings,
        "avg_iris_score": avg_iris_score,
        "avg_directional_score": avg_directional_score,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
    }


def plot_sentiment_donut(sent_df: pd.DataFrame):
    """HELIOS-styled donut chart for the FinBERT sentiment mix."""
    counts = sent_df["Sentiment"].astype(str).str.lower().value_counts()
    labels = ["positive", "neutral", "negative"]
    values = [int(counts.get(label, 0)) for label in labels]

    # Match the black + purple HELIOS theme instead of Matplotlib defaults.
    bg_color = "#000000"
    text_color = "#FFFFFF"
    muted_text = "#D8C4FF"
    accent = "#B266FF"
    colors = ["#B266FF", "#3B0A45", "#FF4B9B"]

    fig, ax = plt.subplots(figsize=(5.4, 5.4), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    if sum(values) == 0:
        pie_values = [1]
        pie_labels = ["No Articles"]
        pie_colors = ["#2A2A2A"]
    else:
        pie_values = values
        pie_labels = [label.title() for label in labels]
        pie_colors = colors

    wedges, texts, autotexts = ax.pie(
        pie_values,
        labels=pie_labels,
        colors=pie_colors,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 0 and sum(values) > 0 else "",
        startangle=90,
        pctdistance=0.76,
        labeldistance=1.08,
        textprops={"color": muted_text, "fontsize": 10, "fontfamily": "DejaVu Sans", "fontweight": "bold"},
        wedgeprops={"width": 0.38, "edgecolor": bg_color, "linewidth": 2.2},
    )

    for autotext in autotexts:
        autotext.set_color(text_color)
        autotext.set_fontsize(10)
        autotext.set_fontfamily("DejaVu Sans")
        autotext.set_fontweight("bold")

    avg_score = float(sent_df["Score"].mean()) if not sent_df.empty else 0.0
    avg_dir = float(sent_df["Directional Score"].mean()) if not sent_df.empty else 0.0
    ax.text(
        0,
        0.08,
        "AVG SCORE",
        ha="center",
        va="center",
        fontsize=10,
        color=muted_text,
        fontfamily="DejaVu Sans",
        fontweight="bold",
    )
    ax.text(
        0,
        -0.07,
        f"{avg_score:.2f}",
        ha="center",
        va="center",
        fontsize=22,
        color=accent,
        fontfamily="DejaVu Sans",
        fontweight="bold",
    )
    ax.text(
        0,
        -0.25,
        f"Bias {avg_dir:.2f}",
        ha="center",
        va="center",
        fontsize=9,
        color=text_color,
        fontfamily="DejaVu Sans",
    )
    ax.axis("equal")
    ax.set_title(
        "Sentiment Article Mix",
        color=accent,
        fontsize=14,
        fontweight="bold",
        pad=18,
        fontfamily="DejaVu Sans",
    )
    fig.patch.set_facecolor(bg_color)
    fig.tight_layout(pad=1.2)
    return fig


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def plot_price_chart(indicators: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    view = indicators.tail(252)
    ax.plot(view.index, view["Close"], label="Close", linewidth=1.8)
    for col in ["SMA_20", "SMA_50", "SMA_200"]:
        if col in view.columns:
            ax.plot(view.index, view[col], label=col, linewidth=1.1)
    ax.set_title(f"{ticker} Price + Moving Averages")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)


def plot_vol_chart(indicators: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    view = indicators.tail(252)
    for col in ["RV_20_%", "RV_60_%"]:
        if col in view.columns:
            ax.plot(view.index, view[col], label=col, linewidth=1.4)
    ax.set_title(f"{ticker} Realized Volatility")
    ax.set_ylabel("Annualized volatility (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)


def display_regime_card(ticker: str, regime_info: Dict[str, object]) -> None:
    st.markdown(
        f"""
        <div class="jxs-card">
        <h3>{ticker} Regime: {regime_info['regime']}</h3>
        <p><b>Suggested paper setup:</b> {regime_info['suggested_setup']}</p>
        <p><b>Reason:</b> {regime_info['reason']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last Price", format_money(regime_info.get("ticker_price")))
    c2.metric("ATM IV", f"{safe_float(regime_info.get('atm_iv'), np.nan) * 100:.1f}%" if np.isfinite(safe_float(regime_info.get("atm_iv"), np.nan)) else "N/A")
    c3.metric("IV Rank Proxy", f"{safe_float(regime_info.get('iv_rank_proxy'), np.nan):.0f}" if np.isfinite(safe_float(regime_info.get("iv_rank_proxy"), np.nan)) else "N/A")
    c4.metric("RV20", f"{safe_float(regime_info.get('rv20'), np.nan):.1f}%" if np.isfinite(safe_float(regime_info.get("rv20"), np.nan)) else "N/A")
    c5.metric("ATR %", f"{safe_float(regime_info.get('atr_pct'), np.nan):.2f}%" if np.isfinite(safe_float(regime_info.get("atr_pct"), np.nan)) else "N/A")

    st.caption("IV Rank Proxy is not true historical IV rank. It compares current ATM option IV against the recent realized-volatility range because free Yahoo chains do not provide a clean historical IV series.")


def format_spread_table(spreads: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    if spreads.empty:
        return spreads
    cols = [
        "candidate_id",
        "final_score",
        "model_score",
        "rule_score",
        "ticker",
        "regime",
        "strategy",
        "expiration",
        "dte",
        "entry_type",
        "premium",
        "width",
        "max_profit_$",
        "max_loss_$",
        "reward_to_risk",
        "short_delta_abs",
        "net_delta",
        "net_theta",
        "net_vega",
        "liquidity_score",
        "avg_bid_ask_pct",
        "iv_rank_proxy",
        "strikes",
        "reason",
    ]
    available = [c for c in cols if c in spreads.columns]
    table = spreads[available].copy()
    if "final_score" in table.columns:
        table["final_score"] = pd.to_numeric(table["final_score"], errors="coerce")
        table = table.sort_values("final_score", ascending=False, na_position="last")
    return table.head(n)


def style_spread_table(spread_table: pd.DataFrame):
    """Format scanner output and highlight the highest final score in HELIOS green."""
    if spread_table.empty:
        return spread_table

    numeric_formats = {
        "final_score": "{:.1f}",
        "model_score": "{:.1f}",
        "rule_score": "{:.1f}",
        "premium": "{:.2f}",
        "width": "{:.2f}",
        "max_profit_$": "${:,.0f}",
        "max_loss_$": "${:,.0f}",
        "reward_to_risk": "{:.2f}",
        "short_delta_abs": "{:.2f}",
        "net_delta": "{:.2f}",
        "net_theta": "{:.2f}",
        "net_vega": "{:.2f}",
        "liquidity_score": "{:.1f}",
        "avg_bid_ask_pct": "{:.1f}%",
        "iv_rank_proxy": "{:.0f}",
    }
    numeric_formats = {col: fmt for col, fmt in numeric_formats.items() if col in spread_table.columns}

    if "final_score" not in spread_table.columns:
        return spread_table.style.format(numeric_formats, na_rep="-")

    top_score = pd.to_numeric(spread_table["final_score"], errors="coerce").max()

    def highlight_top_final_score(column: pd.Series):
        styles = ["" for _ in column]
        if column.name != "final_score" or not np.isfinite(top_score):
            return styles
        for pos, value in enumerate(pd.to_numeric(column, errors="coerce")):
            if np.isfinite(value) and value == top_score:
                styles[pos] = (
                    "background-color: #0B5D1E; "
                    "color: #FFFFFF; "
                    "font-weight: 800; "
                    "border: 1px solid #2EE66B;"
                )
        return styles

    return spread_table.style.format(numeric_formats, na_rep="-").apply(highlight_top_final_score, axis=0)


# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    inject_css()

    st.title("HELIOS Options Matrix")
    st.markdown(
        """
        Named after the personification of the Sun, this dashboard brings clarity to the options market. It combines FinBERT/XGBoost with a new options regime classifier and spread scanner. 
        It does **not** connect to a broker or send orders.
        """
    )

    with st.sidebar:
        st.header("Scanner Inputs")
        ticker_text = st.text_input("Tickers", value="SPY, QQQ, IWM, AAPL", help="Comma-separated. For XSP, Yahoo option chains may be inconsistent; SPY is useful as a liquidity proxy.")
        period = st.selectbox("Price history period", ["6mo", "1y", "2y", "5y"], index=2)
        zero_dte_only = st.checkbox(
            "0DTE iron condor focus",
            value=True,
            help="When enabled, HELIOS prioritizes same-day expirations for iron condor scanning. If no 0DTE chain is available, it falls back to the DTE range below.",
        )
        dte_min, dte_max = st.slider("DTE range", min_value=0, max_value=120, value=(0, 45), step=1)
        min_width, max_width = st.slider("Spread width range", min_value=0.5, max_value=25.0, value=(1.0, 5.0), step=0.5)
        max_bid_ask_pct = st.slider("Max bid-ask spread % per leg", 1, 100, 30)
        min_open_interest = st.number_input("Min open interest per leg", min_value=0, max_value=10000, value=10, step=10)
        min_credit_or_debit = st.number_input("Min credit/debit per share", min_value=0.01, max_value=20.0, value=0.05, step=0.01)
        include_iron_condors = st.checkbox("Build iron condor candidates from verticals", value=True)
        event_risk_days = st.slider("Event-risk window days", 0, 30, 7)
        max_expirations_to_scan = st.slider("Max expirations per ticker", 1, 12, 6)
        run_scan = st.button("Run HELIOS Matrix Scan", type="primary")
        if zero_dte_only:
            st.caption("0DTE mode is ON: same-day expirations will be scanned first when available. Use paper trading only and watch gamma/pin risk.")

    tickers = [clean_ticker(t) for t in ticker_text.split(",") if clean_ticker(t)]

    # Load journal early so ML ranker can be used during scanning.
    journal = load_journal()
    labeled_journal = add_labels_to_journal(journal) if not journal.empty else journal
    model, feature_cols, model_metrics, model_error = train_xgboost_ranker(labeled_journal) if not labeled_journal.empty else (None, MODEL_FEATURES, None, "No journal yet.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1) Regime Dashboard",
        "2) Option Chain + Greeks",
        "3) Spread Scanner",
        "4) Paper Trade Journal + ML Ranker",
        "5) FinBERT Sentiment",
    ])

    # Store latest scan outputs in session state.
    if "latest_regime_rows" not in st.session_state:
        st.session_state["latest_regime_rows"] = pd.DataFrame()
    if "latest_chains" not in st.session_state:
        st.session_state["latest_chains"] = {}
    if "latest_spreads" not in st.session_state:
        st.session_state["latest_spreads"] = pd.DataFrame()

    if run_scan:
        regime_rows = []
        chain_store: Dict[str, Dict[str, pd.DataFrame]] = {}
        all_spreads = []

        progress = st.progress(0, text="Starting scan...")
        total_steps = max(len(tickers), 1)

        for idx, ticker in enumerate(tickers, start=1):
            progress.progress((idx - 1) / total_steps, text=f"Scanning {ticker}...")
            price_df, price_err = get_price_data(ticker, period=period)
            if price_err or price_df.empty:
                regime_rows.append({
                    "Ticker": ticker,
                    "Regime": "Data unavailable",
                    "Suggested paper setup": "Skip",
                    "Reason": price_err or "No data.",
                })
                continue

            indicators = add_indicators(price_df)
            last_price = safe_float(indicators["Close"].iloc[-1])

            expirations, exp_err = get_option_expirations(ticker)
            selected_exps = []
            if expirations:
                # 0DTE support: Yahoo expirations are date strings, so DTE == 0 means same-day expiration.
                same_day_exps = [e for e in expirations if dte_from_expiration(e) == 0]
                range_exps = [e for e in expirations if dte_min <= dte_from_expiration(e) <= dte_max]

                if zero_dte_only and same_day_exps:
                    selected_exps = same_day_exps[:max_expirations_to_scan]
                else:
                    # If 0DTE focus is on but no same-day chain exists, fall back to the normal DTE range.
                    selected_exps = range_exps[:max_expirations_to_scan]

            # Pull first selected expiration to estimate ATM IV for regime.
            atm_iv = np.nan
            first_calls, first_puts = pd.DataFrame(), pd.DataFrame()
            if selected_exps:
                raw_calls, raw_puts, _ = get_option_chain(ticker, selected_exps[0])
                first_calls, first_puts = raw_calls, raw_puts
                atm_iv = atm_iv_from_chain(raw_calls, raw_puts, last_price)

            iv_rank = iv_rank_proxy(atm_iv, indicators)
            earnings_date = get_calendar_event_date(ticker)
            regime_info = classify_regime(indicators, atm_iv, iv_rank, earnings_date, event_risk_days)

            regime_rows.append({
                "Ticker": ticker,
                "Price": last_price,
                "Regime": regime_info["regime"],
                "Suggested paper setup": regime_info["suggested_setup"],
                "Reason": regime_info["reason"],
                "ATM IV": atm_iv,
                "IV Rank Proxy": iv_rank,
                "RV20 %": regime_info["rv20"],
                "ATR %": regime_info["atr_pct"],
                "Trend Strength": regime_info["trend_strength"],
                "Next Event": earnings_date,
            })

            ticker_chain_parts = {}
            if not selected_exps:
                continue

            for exp in selected_exps:
                raw_calls, raw_puts, chain_err = get_option_chain(ticker, exp)
                if chain_err:
                    continue
                clean_calls = clean_options_dataframe(raw_calls, "call", exp, last_price)
                clean_puts = clean_options_dataframe(raw_puts, "put", exp, last_price)
                ticker_chain_parts[f"{ticker}_{exp}_calls"] = clean_calls
                ticker_chain_parts[f"{ticker}_{exp}_puts"] = clean_puts

                verticals = generate_verticals_for_expiration(
                    ticker=ticker,
                    calls=clean_calls,
                    puts=clean_puts,
                    expiration=exp,
                    underlying_price=last_price,
                    regime_info=regime_info,
                    min_width=min_width,
                    max_width=max_width,
                    max_bid_ask_pct=max_bid_ask_pct,
                    min_open_interest=min_open_interest,
                    min_credit_or_debit=min_credit_or_debit,
                )
                if not verticals.empty:
                    verticals = score_spreads_rule_based(verticals, regime_info["regime"])
                    all_spreads.append(verticals)

            if ticker_chain_parts:
                chain_store[ticker] = ticker_chain_parts

        progress.progress(1.0, text="Scan complete.")

        regime_df = pd.DataFrame(regime_rows)
        st.session_state["latest_regime_rows"] = regime_df
        st.session_state["latest_chains"] = chain_store

        spreads_df = pd.concat(all_spreads, ignore_index=True) if all_spreads else pd.DataFrame()
        if include_iron_condors and not spreads_df.empty:
            # Rule-score first, then build ICs and score those too.
            ic = generate_iron_condor_candidates(spreads_df)
            if not ic.empty:
                ic = score_spreads_rule_based(ic)
                spreads_df = pd.concat([spreads_df, ic], ignore_index=True)

        if not spreads_df.empty:
            spreads_df = apply_ml_ranking(spreads_df, model, feature_cols or MODEL_FEATURES)
            spreads_df = spreads_df.sort_values("final_score", ascending=False).reset_index(drop=True)
            spreads_df["candidate_id"] = np.arange(1, len(spreads_df) + 1)

        st.session_state["latest_spreads"] = spreads_df
        st.success("Scan complete. Open the tabs below to review regime, chains, scanner results, and journal actions.")

    # -------------------------------------------------------------------------
    # TAB 1: REGIME DASHBOARD
    # -------------------------------------------------------------------------
    with tab1:
        st.header("Regime Dashboard")
        st.write("This tab summarizes trend, volatility, IV proxy, and event risk into a paper-trading setup label.")

        regime_df = st.session_state.get("latest_regime_rows", pd.DataFrame())
        if regime_df.empty:
            st.info("Set inputs in the sidebar and click **Run HELIOS Scan**.")
        else:
            display_df = regime_df.copy()
            for col in ["Price", "ATM IV", "IV Rank Proxy", "RV20 %", "ATR %", "Trend Strength"]:
                if col in display_df.columns:
                    display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            selected_ticker = st.selectbox("View detailed chart for ticker", display_df["Ticker"].dropna().unique())
            price_df, err = get_price_data(selected_ticker, period=period)
            if err or price_df.empty:
                st.warning(err)
            else:
                indicators = add_indicators(price_df)
                row = display_df[display_df["Ticker"] == selected_ticker].iloc[0]
                regime_info = {
                    "ticker_price": row.get("Price"),
                    "regime": row.get("Regime"),
                    "suggested_setup": row.get("Suggested paper setup"),
                    "reason": row.get("Reason"),
                    "atm_iv": row.get("ATM IV"),
                    "iv_rank_proxy": row.get("IV Rank Proxy"),
                    "rv20": row.get("RV20 %"),
                    "rv60": safe_float(indicators["RV_60_%"].iloc[-1], np.nan),
                    "atr_pct": row.get("ATR %"),
                    "trend_strength": row.get("Trend Strength"),
                    "direction_bias": "",
                    "earnings_date": row.get("Next Event"),
                }
                display_regime_card(selected_ticker, regime_info)
                c1, c2 = st.columns(2)
                with c1:
                    plot_price_chart(indicators, selected_ticker)
                with c2:
                    plot_vol_chart(indicators, selected_ticker)

    # -------------------------------------------------------------------------
    # TAB 2: OPTION CHAIN + GREEKS
    # -------------------------------------------------------------------------
    with tab2:
        st.header("Option Chain + Approximate Greeks")
        st.write("The app calculates delta, theta, and vega from current option IV using a Black-Scholes approximation.")
        chains = st.session_state.get("latest_chains", {})
        if not chains:
            st.info("Run a scan first to load option chains.")
        else:
            ticker_choice = st.selectbox("Ticker", list(chains.keys()), key="chain_ticker")
            keys = list(chains[ticker_choice].keys())
            chain_key = st.selectbox("Chain", keys, key="chain_key")
            chain_df = chains[ticker_choice][chain_key]
            st.dataframe(
                chain_df[[
                    "contractSymbol", "option_type", "expiration", "dte", "strike", "bid", "ask", "mid",
                    "bid_ask_pct", "volume", "openInterest", "impliedVolatility", "delta", "theta", "vega", "liquidity_score"
                ]].sort_values("strike"),
                use_container_width=True,
                hide_index=True,
            )

    # -------------------------------------------------------------------------
    # TAB 3: SPREAD SCANNER
    # -------------------------------------------------------------------------
    with tab3:
        st.header("Spread Scanner")
        st.write("Ranks vertical spreads, and optionally builds iron-condor candidates from matched put/call credit spreads. 0DTE iron condors are supported when same-day expirations are available.")

        spreads_df = st.session_state.get("latest_spreads", pd.DataFrame())
        if spreads_df.empty:
            st.info("Run a scan first. If you already scanned and this is empty, loosen filters such as bid-ask %, open interest, width, or DTE.")
        else:
            strategy_filter = st.multiselect(
                "Filter by strategy",
                options=sorted(spreads_df["strategy"].dropna().unique()),
                default=list(sorted(spreads_df["strategy"].dropna().unique())),
            )
            min_score = st.slider("Minimum final score", 0, 100, 0)
            filtered = spreads_df[
                spreads_df["strategy"].isin(strategy_filter)
                & (pd.to_numeric(spreads_df["final_score"], errors="coerce").fillna(0) >= min_score)
            ].copy()
            filtered["final_score"] = pd.to_numeric(filtered["final_score"], errors="coerce")
            filtered = filtered.sort_values("final_score", ascending=False, na_position="last").reset_index(drop=True)

            spread_table = format_spread_table(filtered, n=150)
            st.dataframe(style_spread_table(spread_table), use_container_width=True, hide_index=True)

            st.download_button(
                "Download scanner results CSV",
                data=filtered.to_csv(index=False),
                file_name="jxs_spread_scanner_results.csv",
                mime="text/csv",
            )

            st.subheader("Add Candidate to Paper Journal")
            candidate_ids = list(filtered["candidate_id"].astype(int)) if not filtered.empty else []
            if candidate_ids:
                chosen_id = st.selectbox("Candidate ID", candidate_ids)
                notes = st.text_area("Mistake / plan notes before entry", placeholder="Example: Enter only if fill is near mid; avoid if price breaches short strike before entry.")
                if st.button("Add selected candidate to paper journal"):
                    candidate = filtered[filtered["candidate_id"] == chosen_id].iloc[0]
                    current_journal = load_journal()
                    new_row = pd.DataFrame([journal_row_from_candidate(candidate, notes)])
                    updated = pd.concat([current_journal, new_row], ignore_index=True)
                    save_journal(updated)
                    st.success("Added to paper journal. Open the Journal tab to edit exits and labels.")
            else:
                st.warning("No candidate IDs available after filters.")

            with st.expander("How to read the scores"):
                st.markdown(
                    """
                    - **Rule Score** = regime fit + liquidity + reward/risk + DTE + Greek profile.
                    - **Model Score** = XGBoost probability of success from your labeled paper-trade journal, only available after enough labeled trades.
                    - **Final Score** = blended score: mostly ML score when available, otherwise rule score.
                    """
                )

    # -------------------------------------------------------------------------
    # TAB 4: PAPER TRADE JOURNAL + ML RANKER
    # -------------------------------------------------------------------------
    with tab4:
        st.header("Paper Trade Journal + XGBoost Ranker")
        st.warning("This journal is for paper trades only. On Streamlit Cloud, download your CSV regularly because app storage can reset.")

        uploaded = st.file_uploader("Upload existing journal CSV", type=["csv"])
        if uploaded is not None:
            journal = load_journal(uploaded)
            save_journal(journal)
            st.success("Uploaded journal loaded into the app session.")
        else:
            journal = load_journal()

        c1, c2, c3 = st.columns(3)
        c1.metric("Journal Rows", len(journal))
        if not journal.empty:
            labeled_counts = add_labels_to_journal(journal)["label"].value_counts().to_dict()
            c2.metric("Known Success Labels", int(sum(labeled_counts.get(x, 0) for x in ["hit_50_profit", "expired_profitable", "closed_profitable"])))
            c3.metric("Known Loss Labels", int(sum(labeled_counts.get(x, 0) for x in ["hit_max_loss", "closed_loss"])))
        else:
            c2.metric("Known Success Labels", 0)
            c3.metric("Known Loss Labels", 0)

        with st.expander("Manual paper-trade entry", expanded=False):
            with st.form("manual_trade_form"):
                mc1, mc2, mc3 = st.columns(3)
                mticker = mc1.text_input("Ticker", value="SPY")
                mstrategy = mc2.selectbox("Strategy", list(STRATEGY_CODE.keys()))
                mexpiration = mc3.text_input("Expiration", value="2026-06-19")
                mstrikes = st.text_input("Strikes", value="Short 500P / Long 495P")
                mc4, mc5, mc6 = st.columns(3)
                entry_type = mc4.selectbox("Entry type", ["credit", "debit"])
                entry_premium = mc5.number_input("Entry premium per share", min_value=0.0, value=1.00, step=0.01)
                max_loss = mc6.number_input("Max loss per share", min_value=0.0, value=4.00, step=0.01)
                mc7, mc8, mc9 = st.columns(3)
                max_profit = mc7.number_input("Max profit per share", min_value=0.0, value=1.00, step=0.01)
                regime = mc8.selectbox("Regime", list(REGIME_CODE.keys()))
                dte = mc9.number_input("DTE", min_value=0, value=30, step=1)
                notes = st.text_area("Mistake / plan notes")
                submit_manual = st.form_submit_button("Add manual paper trade")
                if submit_manual:
                    row = {
                        col: np.nan for col in JOURNAL_COLUMNS
                    }
                    row.update({
                        "trade_id": str(uuid.uuid4())[:8],
                        "entry_date": today_utc_date().isoformat(),
                        "ticker": clean_ticker(mticker),
                        "strategy": mstrategy,
                        "expiration": mexpiration,
                        "dte": dte,
                        "regime": regime,
                        "direction_bias": "",
                        "strikes": mstrikes,
                        "model_score": np.nan,
                        "rule_score": np.nan,
                        "reason": "manual entry",
                        "entry_type": entry_type,
                        "entry_premium": entry_premium,
                        "max_profit": max_profit,
                        "max_loss": max_loss,
                        "mistake_notes": notes,
                        "label": "unknown",
                        "paper_only": True,
                    })
                    journal = pd.concat([journal, pd.DataFrame([row])], ignore_index=True)
                    save_journal(journal)
                    st.success("Manual paper trade added.")

        if not journal.empty:
            if st.button("Auto-label journal outcomes"):
                journal = add_labels_to_journal(journal)
                save_journal(journal)
                st.success("Journal labels updated from exit premium / P&L / exit reason.")

            edited = st.data_editor(
                journal,
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                key="journal_editor",
            )
            if st.button("Save edited journal"):
                save_journal(edited)
                st.success("Journal saved.")

            st.download_button(
                "Download paper trade journal CSV",
                data=edited.to_csv(index=False),
                file_name="paper_trade_journal.csv",
                mime="text/csv",
            )
        else:
            st.info("No journal yet. Add a scanner candidate or enter a manual paper trade.")

        st.subheader("XGBoost Ranker Status")
        fresh_journal = load_journal()
        fresh_labeled = add_labels_to_journal(fresh_journal) if not fresh_journal.empty else fresh_journal
        trained_model, trained_features, metrics, err = train_xgboost_ranker(fresh_labeled) if not fresh_labeled.empty else (None, MODEL_FEATURES, None, "No journal yet.")
        if err:
            st.info(err)
        else:
            st.success(f"Ranker trained on {metrics['rows']} labeled rows. Holdout/training accuracy: {metrics['accuracy']:.2%}")
            with st.expander("Classification report"):
                st.code(metrics["report"])

    # -------------------------------------------------------------------------
    # TAB 5: FINBERT SENTIMENT
    # -------------------------------------------------------------------------
    with tab5:
        st.header("FinBERT Sentiment Add-On")
        st.write(
            "This section analyzes Yahoo Finance RSS articles with FinBERT, shows article drop-downs, "
            "and converts the daily news readout into a paper-trading spread bias."
        )
        st.caption("Transformer models can be heavy on Streamlit Cloud. If the app is slow, lower the max article count.")

        sent_ticker = clean_ticker(st.text_input("Sentiment ticker", value=tickers[0] if tickers else "SPY"))
        keyword = st.text_input("Keyword filter", value="")
        max_articles = st.slider("Max RSS articles", 5, 50, 20)

        if st.button("Run FinBERT Sentiment"):
            with st.spinner("Loading FinBERT and reading Yahoo Finance RSS..."):
                sent_df, sent_err = fetch_yahoo_rss_sentiment(sent_ticker, keyword, max_articles)

            if sent_err:
                st.error(sent_err)
            elif sent_df.empty:
                st.warning("No matching articles found. Try removing the keyword filter or using a more liquid Yahoo Finance ticker such as SPY or QQQ.")
            else:
                # Store it so reruns do not instantly clear the output.
                st.session_state["latest_sentiment_df"] = sent_df
                st.session_state["latest_sentiment_ticker"] = sent_ticker

        sent_df = st.session_state.get("latest_sentiment_df", pd.DataFrame())
        active_sent_ticker = st.session_state.get("latest_sentiment_ticker", sent_ticker)

        if not sent_df.empty:
            avg_score = float(sent_df["Score"].mean())
            avg_directional = float(sent_df["Directional Score"].mean())
            pos = int((sent_df["Directional Score"] > 0).sum())
            neg = int((sent_df["Directional Score"] < 0).sum())
            neu = int((sent_df["Directional Score"] == 0).sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Articles", len(sent_df))
            c2.metric("Average Sentiment", f"{avg_score:.2f}")
            c3.metric("News Bias Score", f"{avg_directional:.2f}")
            c4.metric("Positive / Neutral / Negative", f"{pos} / {neu} / {neg}")

            # Pull current regime context if the ticker was part of the latest scan.
            regime_df_for_news = st.session_state.get("latest_regime_rows", pd.DataFrame())
            regime_row = None
            if not regime_df_for_news.empty and "Ticker" in regime_df_for_news.columns:
                match = regime_df_for_news[regime_df_for_news["Ticker"].astype(str).str.upper() == active_sent_ticker]
                if not match.empty:
                    regime_row = match.iloc[0]

            readout = build_news_spread_readout(sent_df, regime_row)

            st.markdown("### News-to-Spread Readout")
            chart_col, readout_col = st.columns([1, 1.35])
            with chart_col:
                # Small spacer keeps the chart visually lower than the section title and aligned with the readout card.
                st.markdown("<div style='height: 1.05rem;'></div>", unsafe_allow_html=True)
                st.pyplot(plot_sentiment_donut(sent_df), use_container_width=True)

            with readout_col:
                bias_value = html.escape(str(readout.get("bias", "N/A")))
                risk_value = html.escape(str(readout.get("risk_level", "N/A")))
                st.markdown(
                    f"""
                    <div class="news-readout-metrics">
                        <div class="news-mini-metric">
                            <div class="news-mini-label">News Bias</div>
                            <div class="news-mini-value">{bias_value}</div>
                        </div>
                        <div class="news-mini-metric">
                            <div class="news-mini-label">Volatility Risk</div>
                            <div class="news-mini-value">{risk_value}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(f"**Spread interpretation:** {readout['spread_readout']}")
                st.markdown(f"**Volatility warning:** {readout['volatility_warning']}")

                if readout["warnings"]:
                    st.warning(" | ".join(readout["warnings"]))
                else:
                    st.success("No major earnings/macro/legal/volatility flags detected in the matched article set.")

                if regime_row is not None:
                    st.caption(
                        f"Using latest scanner context for {active_sent_ticker}: "
                        f"Regime = {regime_row.get('Regime', 'N/A')}, "
                        f"IV Rank Proxy = {safe_float(regime_row.get('IV Rank Proxy'), np.nan):.0f}"
                    )
                else:
                    st.caption("Run the HELIOS Matrix Scan for this ticker first if you want the news readout to include regime + IV context.")

            st.write(f"### All Analyzed Articles ({len(sent_df)} total)")
            for idx, article in sent_df.reset_index(drop=True).iterrows():
                with st.expander(f"Article {idx + 1}: {article['Title']}"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**Date:** {article['Date']}")
                        st.write(f"**Sentiment:** {article['Sentiment']}")
                        st.write(f"**Score:** {safe_float(article['Score'], 0):.2f}")
                        st.write(f"**News Bias Score:** {safe_float(article['Directional Score'], 0):.2f}")
                        st.write(f"**Risk Flags:** {article.get('Risk Flags', 'None detected')}")
                        if article["Link"]:
                            st.write(f"[Read Full Article]({article['Link']})")
                    with col2:
                        st.write("**Summary:**")
                        st.write(article["Full Text"])


    st.markdown("---")
    st.markdown(
        """
        <p class="small-note">
        Educational/paper-trading dashboard only. Options involve risk; this app estimates Greeks and ranks setups from incomplete public data and should not be treated as live execution advice.
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()