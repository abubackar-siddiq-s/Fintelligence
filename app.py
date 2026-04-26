import streamlit as st
import yfinance as yf
import google.generativeai as genai
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ENV & MODEL SETUP
# ============================================================

load_dotenv()
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FINTELLIGENCE — Stock Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS — DARK TERMINAL LUXURY AESTHETIC
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background: #080B0F !important;
    color: #C8D4DC !important;
}

.stApp {
    background: #080B0F !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,210,120,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 90% 80%, rgba(0,140,255,0.05) 0%, transparent 55%) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #1F2D22; border-radius: 4px; }

/* ── HEADER ── */
.apex-header {
    text-align: center;
    padding: 48px 0 20px;
}
.apex-logo {
    font-family: 'Syne', sans-serif;
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -3px;
    background: linear-gradient(135deg, #00E87A 0%, #00BFFF 60%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 8px;
}
.apex-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 6px;
    color: #3D5040;
    text-transform: uppercase;
    margin-bottom: 40px;
}

/* ── INPUT SECTION ── */
.input-shell {
    background: rgba(13,17,12,0.8);
    border: 1px solid rgba(0,232,122,0.12);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 32px;
    backdrop-filter: blur(24px);
}

.stTextInput > div > div > input {
    background: rgba(5,8,6,0.9) !important;
    border: 1px solid rgba(0,232,122,0.2) !important;
    border-radius: 10px !important;
    color: #00E87A !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 500 !important;
    letter-spacing: 3px !important;
    padding: 18px 22px !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}
.stTextInput > div > div > input::placeholder {
    color: #2A3D2D !important;
    font-size: 14px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
.stTextInput > div > div > input:focus {
    border: 1px solid #00E87A !important;
    box-shadow: 0 0 0 3px rgba(0,232,122,0.08), 0 0 30px rgba(0,232,122,0.06) !important;
    outline: none !important;
}

/* ── EXCHANGE RADIO ── */
.stRadio > div {
    flex-direction: row !important;
    gap: 12px !important;
}
.stRadio > div > label {
    background: rgba(13,20,14,0.8) !important;
    border: 1px solid rgba(0,232,122,0.15) !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    cursor: pointer !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    color: #5A7A5E !important;
    transition: all 0.2s !important;
}
.stRadio > div > label:hover {
    border-color: rgba(0,232,122,0.4) !important;
    color: #00E87A !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #00E87A, #00BFFF) !important;
    color: #050A06 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 40px !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 40px rgba(0,232,122,0.3) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active {
    transform: scale(0.97) !important;
}

/* ── METRIC TILES ── */
div[data-testid="metric-container"] {
    background: rgba(10,15,11,0.9) !important;
    border: 1px solid rgba(0,232,122,0.1) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(0,232,122,0.35) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 30px rgba(0,0,0,0.5) !important;
}
div[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    color: #3D5040 !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #E0EDE2 !important;
}

/* ── SECTION CARD ── */
.section-card {
    background: rgba(10,15,11,0.85);
    border: 1px solid rgba(0,232,122,0.09);
    border-radius: 18px;
    padding: 30px 34px;
    margin-bottom: 24px;
    backdrop-filter: blur(20px);
    transition: border-color 0.3s;
}
.section-card:hover {
    border-color: rgba(0,232,122,0.2);
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 5px;
    text-transform: uppercase;
    color: #3D6040;
    margin-bottom: 22px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(0,232,122,0.07);
}
.section-body {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.85;
    color: #A8C4AC;
    white-space: pre-wrap;
}

/* ── SCORE BADGE ── */
.score-wrap {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 20px 0;
}
.score-badge {
    font-family: 'Syne', sans-serif;
    font-size: 52px;
    font-weight: 800;
    line-height: 1;
}
.score-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3D5040;
    margin-top: 4px;
}
.score-green  { color: #00E87A; }
.score-yellow { color: #F5C842; }
.score-red    { color: #FF4D6D; }

/* ── SIGNAL TAG ── */
.signal-tag {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 6px;
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 4px 4px 4px 0;
}
.tag-buy    { background: rgba(0,232,122,0.12); color: #00E87A; border: 1px solid rgba(0,232,122,0.3); }
.tag-sell   { background: rgba(255,77,109,0.12); color: #FF4D6D; border: 1px solid rgba(255,77,109,0.3); }
.tag-hold   { background: rgba(245,200,66,0.12); color: #F5C842; border: 1px solid rgba(245,200,66,0.3); }
.tag-watch  { background: rgba(0,191,255,0.12); color: #00BFFF; border: 1px solid rgba(0,191,255,0.3); }
.tag-risk   { background: rgba(255,100,50,0.1); color: #FF8C50; border: 1px solid rgba(255,100,50,0.25); }

/* ── DIVIDER ── */
hr { border-color: rgba(0,232,122,0.06) !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,15,11,0.7) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    border: 1px solid rgba(0,232,122,0.1) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #3D5040 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,232,122,0.1) !important;
    color: #00E87A !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    color: #3D6040 !important;
    background: rgba(10,15,11,0.7) !important;
    border: 1px solid rgba(0,232,122,0.08) !important;
    border-radius: 10px !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #00E87A !important; }

/* ── FOOTER ── */
footer { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ── PROGRESS ── */
.stProgress > div > div { background: linear-gradient(90deg, #00E87A, #00BFFF) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="apex-header">
    <div class="apex-logo">FINTELLIGENCE</div>
    <div class="apex-tagline">Institutional Stock Intelligence Terminal</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT UI
# ============================================================

st.markdown("<div class='input-shell'>", unsafe_allow_html=True)
col_a, col_b = st.columns([3, 1])
with col_a:
    ticker_raw = st.text_input("", placeholder="RELIANCE / TCS / AAPL / TSLA", label_visibility="collapsed")
with col_b:
    exchange = st.radio("Exchange", ["NSE 🇮🇳", "BSE 🇮🇳", "US 🇺🇸", "Custom"], horizontal=True, label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

col_x, col_y, col_z = st.columns([2, 1, 2])
with col_y:
    analyze_btn = st.button("⟡  ANALYZE", use_container_width=True)

# ============================================================
# TICKER RESOLUTION
# ============================================================

def resolve_ticker(raw, exch):
    raw = raw.strip().upper()
    if not raw:
        return None
    if exch == "NSE 🇮🇳":
        return raw + ".NS"
    elif exch == "BSE 🇮🇳":
        return raw + ".BO"
    elif exch == "US 🇺🇸":
        return raw
    else:
        return raw  # Custom / already has suffix

# ============================================================
# DATA FETCHERS
# ============================================================

@st.cache_data(ttl=300)
def fetch_info(ticker):
    stk = yf.Ticker(ticker)
    return stk.info

@st.cache_data(ttl=300)
def fetch_history(ticker, period="1y"):
    stk = yf.Ticker(ticker)
    return stk.history(period=period)

@st.cache_data(ttl=300)
def fetch_financials(ticker):
    stk = yf.Ticker(ticker)
    return {
        "income": stk.financials,
        "balance": stk.balance_sheet,
        "cashflow": stk.cashflow,
        "quarterly_income": stk.quarterly_financials,
        "quarterly_balance": stk.quarterly_balance_sheet,
    }

@st.cache_data(ttl=300)
def fetch_holders(ticker):
    stk = yf.Ticker(ticker)
    try:
        inst = stk.institutional_holders
        major = stk.major_holders
    except:
        inst, major = None, None
    return inst, major

@st.cache_data(ttl=300)
def fetch_earnings(ticker):
    stk = yf.Ticker(ticker)
    try:
        return stk.earnings_dates
    except:
        return None

# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def compute_technicals(df):
    c = df["Close"].copy()
    v = df["Volume"].copy()
    h = df["High"].copy()
    lo = df["Low"].copy()

    def ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def sma(s, n): return s.rolling(n).mean()

    out = {}

    # Moving Averages
    out["SMA_20"]  = sma(c, 20)
    out["SMA_50"]  = sma(c, 50)
    out["SMA_200"] = sma(c, 200)
    out["EMA_9"]   = ema(c, 9)
    out["EMA_21"]  = ema(c, 21)

    # MACD
    ema12 = ema(c, 12); ema26 = ema(c, 26)
    out["MACD"]        = ema12 - ema26
    out["MACD_signal"] = ema(out["MACD"], 9)
    out["MACD_hist"]   = out["MACD"] - out["MACD_signal"]

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    mid = sma(c, 20)
    std = c.rolling(20).std()
    out["BB_mid"]   = mid
    out["BB_upper"] = mid + 2 * std
    out["BB_lower"] = mid - 2 * std

    # Stochastic
    low14  = lo.rolling(14).min()
    high14 = h.rolling(14).max()
    out["%K"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    out["%D"] = out["%K"].rolling(3).mean()

    # ATR
    tr = pd.concat([
        h - lo,
        (h - c.shift()).abs(),
        (lo - c.shift()).abs()
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()

    # OBV
    obv = (v * np.sign(c.diff())).fillna(0).cumsum()
    out["OBV"] = obv

    # VWAP (rolling)
    tp = (h + lo + c) / 3
    out["VWAP"] = (tp * v).cumsum() / v.cumsum()

    # ADX
    plus_dm  = (h.diff()).clip(lower=0)
    minus_dm = (-lo.diff()).clip(lower=0)
    plus_di  = 100 * ema(plus_dm, 14) / out["ATR"].replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, 14) / out["ATR"].replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["ADX"]       = dx.rolling(14).mean()
    out["Plus_DI"]   = plus_di
    out["Minus_DI"]  = minus_di

    # CCI
    out["CCI"] = (tp - sma(tp, 20)) / (0.015 * tp.rolling(20).std())

    # Williams %R
    out["Williams_R"] = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    return out

def interpret_technicals(df, indicators, info):
    last = df.index[-1]
    price = df["Close"].iloc[-1]

    rsi     = indicators["RSI"].iloc[-1]
    macd    = indicators["MACD"].iloc[-1]
    macd_s  = indicators["MACD_signal"].iloc[-1]
    adx     = indicators["ADX"].iloc[-1]
    cci     = indicators["CCI"].iloc[-1]
    wr      = indicators["Williams_R"].iloc[-1]
    stk_k   = indicators["%K"].iloc[-1]
    stk_d   = indicators["%D"].iloc[-1]
    sma20   = indicators["SMA_20"].iloc[-1]
    sma50   = indicators["SMA_50"].iloc[-1]
    sma200  = indicators["SMA_200"].iloc[-1]
    bb_u    = indicators["BB_upper"].iloc[-1]
    bb_l    = indicators["BB_lower"].iloc[-1]
    bb_m    = indicators["BB_mid"].iloc[-1]
    atr     = indicators["ATR"].iloc[-1]
    vwap    = indicators["VWAP"].iloc[-1]
    obv_now = indicators["OBV"].iloc[-1]
    obv_prev= indicators["OBV"].iloc[-20]

    signals = []
    bullish = 0; bearish = 0

    # RSI
    if rsi < 30:   signals.append(("RSI Oversold", "BUY"));  bullish += 2
    elif rsi < 45: signals.append(("RSI Weak",     "WATCH")); bullish += 1
    elif rsi > 70: signals.append(("RSI Overbought","SELL")); bearish += 2
    elif rsi > 55: signals.append(("RSI Strong",   "HOLD"));  bullish += 1
    else:          signals.append(("RSI Neutral",  "HOLD"))

    # MACD
    if macd > macd_s:   signals.append(("MACD Bullish Cross", "BUY"));  bullish += 2
    else:               signals.append(("MACD Bearish Cross", "SELL")); bearish += 2

    # Trend vs MAs
    if price > sma20 > sma50 > sma200:
        signals.append(("Golden Alignment ✦", "BUY")); bullish += 3
    elif price < sma20 < sma50 < sma200:
        signals.append(("Death Alignment ✦", "SELL")); bearish += 3
    elif price > sma200:
        signals.append(("Above 200 SMA", "HOLD")); bullish += 1
    else:
        signals.append(("Below 200 SMA", "SELL")); bearish += 1

    # ADX Trend Strength
    if adx > 25:   signals.append((f"Trend Strong (ADX={adx:.0f})", "HOLD")); bullish += 1
    elif adx < 15: signals.append((f"No Trend (ADX={adx:.0f})", "WATCH"))

    # Stochastic
    if stk_k < 20 and stk_k > stk_d: signals.append(("Stoch Oversold Cross","BUY")); bullish += 2
    elif stk_k > 80:                  signals.append(("Stoch Overbought","SELL")); bearish += 1

    # Bollinger
    bb_pos = (price - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5
    if bb_pos < 0.15: signals.append(("BB Lower Band Touch","BUY")); bullish += 2
    elif bb_pos > 0.85: signals.append(("BB Upper Band Touch","SELL")); bearish += 1

    # OBV
    if obv_now > obv_prev: signals.append(("OBV Rising (Smart Money)","BUY")); bullish += 1
    else:                  signals.append(("OBV Falling","SELL")); bearish += 1

    # VWAP
    if price > vwap: signals.append(("Above VWAP","BUY")); bullish += 1
    else:            signals.append(("Below VWAP","SELL")); bearish += 1

    # CCI
    if cci < -100: signals.append(("CCI Oversold","BUY")); bullish += 1
    elif cci > 100: signals.append(("CCI Overbought","SELL")); bearish += 1

    total = bullish + bearish
    score = int((bullish / total) * 100) if total > 0 else 50

    if score >= 70:   verdict = "STRONG BUY"
    elif score >= 55: verdict = "BUY"
    elif score >= 45: verdict = "HOLD"
    elif score >= 30: verdict = "SELL"
    else:             verdict = "STRONG SELL"

    # Support / Resistance (pivot points)
    h_last = df["High"].iloc[-5:].max()
    l_last = df["Low"].iloc[-5:].min()
    pivot  = (h_last + l_last + price) / 3
    r1 = 2 * pivot - l_last
    s1 = 2 * pivot - h_last
    r2 = pivot + (h_last - l_last)
    s2 = pivot - (h_last - l_last)

    # Stop Loss & Targets
    stop  = round(price - 1.5 * atr, 2)
    tgt1  = round(price + 2 * atr, 2)
    tgt2  = round(price + 4 * atr, 2)
    tgt3  = round(price + 7 * atr, 2)
    risk_reward = round((tgt1 - price) / (price - stop), 2) if price != stop else 0

    return {
        "signals": signals,
        "bullish": bullish,
        "bearish": bearish,
        "score": score,
        "verdict": verdict,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_s,
        "adx": adx,
        "atr": atr,
        "pivot": pivot,
        "r1": r1, "r2": r2,
        "s1": s1, "s2": s2,
        "stop": stop,
        "tgt1": tgt1, "tgt2": tgt2, "tgt3": tgt3,
        "rr": risk_reward,
        "bb_pos": bb_pos,
        "price": price,
        "vwap": vwap,
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
    }

# ============================================================
# FUNDAMENTAL ANALYSIS
# ============================================================

def compute_fundamentals(info, fin):
    scores = {}

    # Valuation
    pe    = info.get("trailingPE") or 0
    pb    = info.get("priceToBook") or 0
    ps    = info.get("priceToSalesTrailingTwelveMonths") or 0
    peg   = info.get("pegRatio") or 0
    ev_eb = info.get("enterpriseToEbitda") or 0

    val_score = 50
    if 0 < pe < 15:   val_score += 20
    elif 15 <= pe < 25: val_score += 10
    elif pe > 40:     val_score -= 15
    if 0 < pb < 2:    val_score += 10
    if 0 < peg < 1:   val_score += 15
    elif peg > 2:     val_score -= 10
    scores["Valuation"] = max(0, min(100, val_score))

    # Profitability
    roe    = (info.get("returnOnEquity") or 0) * 100
    roa    = (info.get("returnOnAssets") or 0) * 100
    npm    = (info.get("profitMargins") or 0) * 100
    opm    = (info.get("operatingMargins") or 0) * 100
    gpm    = (info.get("grossMargins") or 0) * 100

    prof_score = 50
    if roe > 20: prof_score += 25
    elif roe > 12: prof_score += 12
    elif roe < 5: prof_score -= 15
    if npm > 15: prof_score += 15
    elif npm > 8: prof_score += 7
    if opm > 20: prof_score += 10
    scores["Profitability"] = max(0, min(100, prof_score))

    # Financial Health
    cr     = info.get("currentRatio") or 0
    de     = info.get("debtToEquity") or 0
    ic     = info.get("interestCoverage") or 0
    qr     = info.get("quickRatio") or 0

    health_score = 50
    if 1.5 <= cr <= 3: health_score += 20
    elif cr < 1: health_score -= 20
    if de < 0.3: health_score += 20
    elif de < 1: health_score += 10
    elif de > 2: health_score -= 15
    scores["Financial Health"] = max(0, min(100, health_score))

    # Growth
    rev_g  = (info.get("revenueGrowth") or 0) * 100
    earn_g = (info.get("earningsGrowth") or 0) * 100
    fwd_pe = info.get("forwardPE") or 0

    growth_score = 50
    if rev_g > 20: growth_score += 25
    elif rev_g > 10: growth_score += 12
    elif rev_g < 0: growth_score -= 20
    if earn_g > 25: growth_score += 20
    elif earn_g > 10: growth_score += 10
    elif earn_g < 0: growth_score -= 15
    scores["Growth"] = max(0, min(100, growth_score))

    # Moat & Quality
    beta   = info.get("beta") or 1
    div_y  = (info.get("dividendYield") or 0) * 100
    insider_pct = (info.get("heldPercentInsiders") or 0) * 100
    inst_pct    = (info.get("heldPercentInstitutions") or 0) * 100

    moat_score = 50
    if npm > 20: moat_score += 20
    if roe > 20: moat_score += 15
    if insider_pct > 10: moat_score += 10
    if div_y > 1: moat_score += 5
    scores["Moat & Quality"] = max(0, min(100, moat_score))

    overall = int(np.mean(list(scores.values())))

    if overall >= 75:   verdict = "STRONG BUY"
    elif overall >= 60: verdict = "BUY"
    elif overall >= 45: verdict = "HOLD"
    elif overall >= 30: verdict = "SELL"
    else:               verdict = "STRONG SELL"

    return {
        "scores": scores,
        "overall": overall,
        "verdict": verdict,
        "pe": pe, "pb": pb, "ps": ps, "peg": peg,
        "ev_eb": ev_eb, "fwd_pe": fwd_pe,
        "roe": roe, "roa": roa,
        "npm": npm, "opm": opm, "gpm": gpm,
        "cr": cr, "de": de, "ic": ic,
        "rev_g": rev_g, "earn_g": earn_g,
        "beta": beta, "div_y": div_y,
        "insider_pct": insider_pct, "inst_pct": inst_pct,
    }

# ============================================================
# PLOTLY CHARTS
# ============================================================

CHART_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "font_color":    "#5A7A5E",
    "grid_color":    "rgba(0,232,122,0.04)",
    "green":  "#00E87A",
    "red":    "#FF4D6D",
    "blue":   "#00BFFF",
    "yellow": "#F5C842",
    "purple": "#A78BFA",
}

def chart_candlestick(df, indicators, ticker_symbol):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.17, 0.17, 0.16],
        vertical_spacing=0.02,
        subplot_titles=["", "MACD", "RSI", "Volume"]
    )
    T = CHART_THEME

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_fillcolor=T["green"], increasing_line_color=T["green"],
        decreasing_fillcolor=T["red"],   decreasing_line_color=T["red"],
        name="Price", showlegend=False
    ), row=1, col=1)

    # MAs
    for ma, col, w in [("SMA_20", T["yellow"], 1.2), ("SMA_50", T["blue"], 1.2), ("SMA_200", T["purple"], 1.5)]:
        fig.add_trace(go.Scatter(
            x=df.index, y=indicators[ma],
            mode="lines", line=dict(color=col, width=w),
            name=ma
        ), row=1, col=1)

    # BB
    fig.add_trace(go.Scatter(
        x=df.index, y=indicators["BB_upper"],
        line=dict(color="rgba(0,232,122,0.2)", width=1, dash="dot"),
        name="BB Upper", showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=indicators["BB_lower"],
        fill="tonexty", fillcolor="rgba(0,232,122,0.03)",
        line=dict(color="rgba(0,232,122,0.2)", width=1, dash="dot"),
        name="BB Lower", showlegend=False
    ), row=1, col=1)

    # VWAP
    fig.add_trace(go.Scatter(
        x=df.index, y=indicators["VWAP"],
        mode="lines", line=dict(color="#FF8C50", width=1.2, dash="dash"),
        name="VWAP"
    ), row=1, col=1)

    # MACD
    colors = [T["green"] if v >= 0 else T["red"] for v in indicators["MACD_hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=indicators["MACD_hist"],
        marker_color=colors, name="MACD Hist", showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=indicators["MACD"],
        line=dict(color=T["blue"], width=1.2), name="MACD"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=indicators["MACD_signal"],
        line=dict(color=T["yellow"], width=1.2), name="Signal"
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=indicators["RSI"],
        line=dict(color=T["purple"], width=1.5), name="RSI"
    ), row=3, col=1)
    for lvl, clr in [(70, T["red"]), (30, T["green"])]:
        fig.add_hline(y=lvl, line_dash="dot",
                    line_color=(
                        f"rgba({','.join(str(int(clr.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.4)"
                        if clr.startswith('#')
                        else clr
                    ),
                      row=3, col=1)

    # Volume
    vcols = [T["green"] if df["Close"].iloc[i] >= df["Open"].iloc[i] else T["red"]
             for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vcols, showlegend=False, name="Volume"
    ), row=4, col=1)

    fig.update_layout(
        paper_bgcolor=T["paper_bgcolor"],
        plot_bgcolor=T["plot_bgcolor"],
        font=dict(family="JetBrains Mono, monospace", color=T["font_color"], size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", y=1.02, x=0,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)"
        ),
        height=780,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(10,15,11,0.95)",
            font=dict(family="JetBrains Mono", size=11, color="#00E87A"),
            bordercolor="rgba(0,232,122,0.2)"
        ),
    )
    for row in [1, 2, 3, 4]:
        fig.update_xaxes(
            showgrid=True, gridcolor=T["grid_color"],
            zeroline=False, row=row, col=1
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=T["grid_color"],
            zeroline=False, row=row, col=1
        )

    return fig


def chart_fundamentals(fund):
    T = CHART_THEME

    # Spider / Radar chart
    cats = list(fund["scores"].keys())
    vals = list(fund["scores"].values())
    cats_closed = cats + [cats[0]]
    vals_closed  = vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed,
        fill="toself",
        fillcolor="rgba(0,232,122,0.08)",
        line=dict(color=T["green"], width=2),
        marker=dict(color=T["green"], size=7),
        name="Score"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=9, color=T["font_color"]),
                gridcolor=T["grid_color"],
                linecolor=T["grid_color"],
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="#A8C4AC"),
                gridcolor=T["grid_color"],
                linecolor=T["grid_color"],
            )
        ),
        paper_bgcolor=T["paper_bgcolor"],
        plot_bgcolor=T["plot_bgcolor"],
        font=dict(family="JetBrains Mono", color=T["font_color"]),
        showlegend=False,
        height=380,
        margin=dict(l=30, r=30, t=30, b=30)
    )
    return fig


def chart_price_history(df):
    T = CHART_THEME

    # Gradient area chart for 1yr
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(0,232,122,0.05)",
        line=dict(color=T["green"], width=2),
        name="Close"
    ))
    fig.update_layout(
        paper_bgcolor=T["paper_bgcolor"],
        plot_bgcolor=T["plot_bgcolor"],
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=T["grid_color"], zeroline=False,
                   tickfont=dict(size=9, color=T["font_color"])),
        hovermode="x",
        hoverlabel=dict(bgcolor="rgba(10,15,11,0.95)",
                        font=dict(family="JetBrains Mono", size=11, color=T["green"]))
    )
    return fig


# ============================================================
# GEMINI PROMPTS — DEEP EXPERT ANALYSIS
# ============================================================

def build_deep_fundamental_prompt(ticker, info, fund):
    return f"""
You are a legendary fund manager with 30 years of Wall Street and Dalal Street experience.
Perform DEEP, EXPERT fundamental analysis. Be specific, data-driven, and brutally honest.

=== COMPANY DATA ===
Ticker: {ticker}
Name: {info.get('longName','N/A')}
Sector: {info.get('sector','N/A')} | Industry: {info.get('industry','N/A')}
Description: {str(info.get('longBusinessSummary',''))[:500]}

=== FINANCIAL METRICS ===
Price: {info.get('currentPrice') or info.get('regularMarketPrice')}
P/E (TTM): {fund['pe']:.1f} | Forward P/E: {fund['fwd_pe']:.1f} | PEG: {fund['peg']:.2f}
P/B: {fund['pb']:.2f} | P/S: {fund['ps']:.2f} | EV/EBITDA: {fund['ev_eb']:.1f}
ROE: {fund['roe']:.1f}% | ROA: {fund['roa']:.1f}%
Net Margin: {fund['npm']:.1f}% | Operating Margin: {fund['opm']:.1f}% | Gross Margin: {fund['gpm']:.1f}%
Revenue Growth (YoY): {fund['rev_g']:.1f}% | Earnings Growth: {fund['earn_g']:.1f}%
Current Ratio: {fund['cr']:.2f} | D/E Ratio: {fund['de']:.2f}
Dividend Yield: {fund['div_y']:.2f}%
Beta: {fund['beta']:.2f}
Insider Holding: {fund['insider_pct']:.1f}% | Institutional: {fund['inst_pct']:.1f}%

=== QUANT SCORES ===
Valuation: {fund['scores']['Valuation']}/100
Profitability: {fund['scores']['Profitability']}/100
Financial Health: {fund['scores']['Financial Health']}/100
Growth: {fund['scores']['Growth']}/100
Moat & Quality: {fund['scores']['Moat & Quality']}/100
Overall Fundamental Score: {fund['overall']}/100

Output this EXACT structure (use emojis exactly as shown):

🏢 BUSINESS QUALITY ASSESSMENT
[3-4 sentences on business model, competitive position, management quality, industry tailwinds/headwinds]

💰 FINANCIAL STRENGTH DEEP DIVE
Profitability: [comment on margins vs industry, trend]
Balance Sheet: [debt quality, liquidity, financial stability]
Cash Flow: [FCF quality, capital allocation discipline]

🏰 ECONOMIC MOAT ANALYSIS
Type: [Brand / Network Effects / Switching Costs / Cost Advantage / None]
Width: [Wide / Narrow / None]
Evidence: [Specific evidence from the numbers above]

📉 VALUATION VERDICT
Intrinsic Value Assessment: [Overvalued/Fairly Valued/Undervalued and why]
Margin of Safety: [Estimate a % margin of safety or lack thereof]
Compared to Sector: [How does valuation compare]

🚀 MULTIBAGGER POTENTIAL
Probability 2x in 3yrs: [High/Medium/Low + reason]
Probability 5x in 5yrs: [High/Medium/Low + reason]
Key Catalysts: [3 specific upcoming catalysts]

⚠ RISKS — TOP 5
1. [Risk + magnitude]
2. [Risk + magnitude]
3. [Risk + magnitude]
4. [Risk + magnitude]
5. [Risk + magnitude]

🏆 FUNDAMENTAL VERDICT
Rating: {fund['verdict']}
[2-3 sentence investment thesis summary]
Position Sizing Recommendation: [% of portfolio for conservative/moderate/aggressive investor]
"""

def build_deep_technical_prompt(ticker, info, tech, df):
    price = tech["price"]
    trend_52w_high = df["High"].max()
    trend_52w_low  = df["Low"].min()
    pct_from_high  = ((price - trend_52w_high) / trend_52w_high) * 100
    pct_from_low   = ((price - trend_52w_low) / trend_52w_low) * 100

    return f"""
You are a legendary technical analyst — ex-Goldman Sachs, CMT Level 3, 25 years reading charts.
Perform DEEP, EXPERT technical analysis. Be specific with levels. Use institutional thinking.

=== PRICE DATA ===
Ticker: {ticker}
Current Price: {price:.2f}
52W High: {trend_52w_high:.2f} ({pct_from_high:.1f}% from high)
52W Low: {trend_52w_low:.2f} (+{pct_from_low:.1f}% from low)
VWAP: {tech['vwap']:.2f}

=== INDICATORS ===
RSI (14): {tech['rsi']:.1f}
MACD: {tech['macd']:.3f} | Signal: {tech['macd_signal']:.3f}
ADX: {tech['adx']:.1f}
ATR: {tech['atr']:.2f}
SMA 20/50/200: {tech['sma20']:.2f} / {tech['sma50']:.2f} / {tech['sma200']:.2f}
BB Position: {tech['bb_pos']*100:.0f}th percentile

=== LEVELS ===
Pivot: {tech['pivot']:.2f}
R1: {tech['r1']:.2f} | R2: {tech['r2']:.2f}
S1: {tech['s1']:.2f} | S2: {tech['s2']:.2f}
Stop Loss: {tech['stop']:.2f}
Target 1: {tech['tgt1']:.2f} | Target 2: {tech['tgt2']:.2f} | Target 3: {tech['tgt3']:.2f}
Risk:Reward: 1:{tech['rr']}

=== TECHNICAL SIGNALS ===
Bull/Bear count: {tech['bullish']} bullish / {tech['bearish']} bearish
Technical Score: {tech['score']}/100

Output this EXACT structure:

📈 TREND ANALYSIS
Primary Trend: [Bullish/Bearish/Sideways + timeframe context]
Trend Strength: [Strong/Moderate/Weak based on ADX]
Price Structure: [Higher highs/lower lows pattern]
MA Alignment: [Comment on price vs 20/50/200 SMA]

📍 KEY PRICE LEVELS
Immediate Support: [Level + reason]
Major Support: [Level + reason]
Immediate Resistance: [Level + reason]
Major Resistance: [Level + reason]
Critical Level to Watch: [The ONE level that changes everything]

🏦 INSTITUTIONAL FOOTPRINT
Volume Analysis: [What institutional volume patterns suggest]
Smart Money Zones: [Where institutions are likely accumulating/distributing]
VWAP Analysis: [Price vs VWAP and implications]

📊 INDICATOR CONFLUENCE
RSI Reading: [What {tech['rsi']:.0f} means in current trend context]
MACD Analysis: [Momentum direction and quality]
Oscillator Consensus: [Are indicators aligned or diverging?]

⚡ TRADE SETUP — ACTIONABLE
Bias: [Bullish/Bearish/Neutral]
Entry Zone: [Price range for entry]
Stop Loss: {tech['stop']:.2f} [reason]
Target 1: {tech['tgt1']:.2f} (+{((tech['tgt1']-price)/price*100):.1f}%)
Target 2: {tech['tgt2']:.2f} (+{((tech['tgt2']-price)/price*100):.1f}%)
Target 3: {tech['tgt3']:.2f} (+{((tech['tgt3']-price)/price*100):.1f}%)
Risk:Reward: 1:{tech['rr']}
Timeframe: [Swing / Positional / Long-Term]

🎯 TECHNICAL VERDICT
Rating: {tech['verdict']}
[2-3 sentence technical summary]
Key Watchout: [The one thing that would invalidate this setup]
"""

def run_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text if response else "Analysis unavailable."
    except Exception as e:
        return f"⚠ Gemini error: {e}"

# ============================================================
# HELPER: SCORE COLOR CLASS
# ============================================================

def score_class(s):
    if s >= 65: return "score-green"
    elif s >= 45: return "score-yellow"
    else: return "score-red"

def verdict_tag(v):
    v = v.upper()
    if "STRONG BUY" in v:  return f'<span class="signal-tag tag-buy">⬆ {v}</span>'
    elif "BUY" in v:       return f'<span class="signal-tag tag-buy">▲ {v}</span>'
    elif "STRONG SELL" in v: return f'<span class="signal-tag tag-sell">⬇ {v}</span>'
    elif "SELL" in v:      return f'<span class="signal-tag tag-sell">▼ {v}</span>'
    else:                  return f'<span class="signal-tag tag-hold">◆ {v}</span>'

def tag_for_signal(sig_type):
    s = sig_type.upper()
    if "BUY" in s:   return "tag-buy"
    elif "SELL" in s: return "tag-sell"
    elif "WATCH" in s: return "tag-watch"
    else: return "tag-hold"

# ============================================================
# MAIN ANALYSIS FLOW
# ============================================================

if analyze_btn:
    ticker = resolve_ticker(ticker_raw, exchange)
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            info = fetch_info(ticker)
            hist_1y = fetch_history(ticker, "1y")
            hist_6mo = fetch_history(ticker, "6mo")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    price = info.get("currentPrice") or info.get("regularMarketPrice") or (
        hist_1y["Close"].iloc[-1] if not hist_1y.empty else None
    )

    if not price:
        st.error("❌ Invalid ticker or no data available.")
        st.stop()

    # ── HERO METRICS ────────────────────────────────────────

    name    = info.get("longName", ticker)
    mktcap  = info.get("marketCap", 0)
    change  = info.get("regularMarketChangePercent", 0) * 100 if info.get("regularMarketChangePercent") else 0
    vol     = info.get("volume") or info.get("regularMarketVolume") or 0
    avgvol  = info.get("averageVolume") or 1
    sector  = info.get("sector", "N/A")
    country = info.get("country", "")
    currency = info.get("financialCurrency", info.get("currency", ""))
    week52_h = info.get("fiftyTwoWeekHigh", 0)
    week52_l = info.get("fiftyTwoWeekLow", 0)
    beta     = info.get("beta", 0)

    def fmt_cap(v):
        if v >= 1e12: return f"{v/1e12:.2f}T"
        if v >= 1e9:  return f"{v/1e9:.2f}B"
        if v >= 1e6:  return f"{v/1e6:.2f}M"
        return str(v)

    chg_col = "#00E87A" if change >= 0 else "#FF4D6D"
    chg_arrow = "▲" if change >= 0 else "▼"

    st.markdown(f"""
    <div style="
        background: rgba(10,15,11,0.85);
        border: 1px solid rgba(0,232,122,0.12);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    ">
        <div style="font-family:'Syne',sans-serif; font-size:13px; letter-spacing:4px; text-transform:uppercase; color:#3D5040;">
            {sector} · {country} · {currency}
        </div>
        <div style="display:flex; align-items:baseline; gap:20px; flex-wrap:wrap;">
            <span style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800; color:#E0EDE2; letter-spacing:-1px;">{name}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:14px; color:#3D6040; letter-spacing:2px;">{ticker}</span>
        </div>
        <div style="display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;">
            <span style="font-family:'Syne',sans-serif; font-size:48px; font-weight:800; color:#E0EDE2; letter-spacing:-2px;">{price:,.2f}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:20px; color:{chg_col}; font-weight:500;">{chg_arrow} {abs(change):.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Market Cap", fmt_cap(mktcap))
    m2.metric("52W High", f"{week52_h:,.2f}")
    m3.metric("52W Low", f"{week52_l:,.2f}")
    m4.metric("Volume", fmt_cap(vol))
    m5.metric("Avg Volume", fmt_cap(avgvol))
    m6.metric("Beta", f"{beta:.2f}" if beta else "N/A")

    st.markdown("---")

    # ── MINI PRICE CHART ─────────────────────────────────────

    if not hist_1y.empty:
        st.plotly_chart(chart_price_history(hist_1y), use_container_width=True, config={"displayModeBar": False})

    # ── COMPUTE ──────────────────────────────────────────────

    with st.spinner("Computing technical indicators..."):
        indicators = compute_technicals(hist_6mo if not hist_6mo.empty else hist_1y)
        tech = interpret_technicals(hist_6mo if not hist_6mo.empty else hist_1y, indicators, info)

    with st.spinner("Computing fundamental scores..."):
        fin = fetch_financials(ticker)
        fund = compute_fundamentals(info, fin)

    # ── OVERALL VERDICT BANNER ──────────────────────────────

    combined = int((tech["score"] + fund["overall"]) / 2)
    if combined >= 70:   combined_verdict = "STRONG BUY"
    elif combined >= 57: combined_verdict = "BUY"
    elif combined >= 43: combined_verdict = "HOLD"
    elif combined >= 30: combined_verdict = "SELL"
    else:                combined_verdict = "STRONG SELL"

    sc = score_class(combined)

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(0,232,122,0.05) 0%, rgba(0,191,255,0.03) 100%);
        border: 1px solid rgba(0,232,122,0.2);
        border-radius: 18px;
        padding: 28px 36px;
        margin: 16px 0 28px;
        display: flex;
        align-items: center;
        gap: 32px;
        flex-wrap: wrap;
    ">
        <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:3px; color:#3D5040; text-transform:uppercase; margin-bottom:6px;">FINTELLIGENCE COMPOSITE VERDICT</div>
            <div style="font-family:'Syne',sans-serif; font-size:42px; font-weight:800;" class="{sc}">{combined_verdict}</div>
        </div>
        <div style="flex:1; display:flex; gap:24px; flex-wrap:wrap;">
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:2px; color:#3D5040; text-transform:uppercase;">Technical Score</div>
                <div style="font-family:'Syne',sans-serif; font-size:32px; font-weight:700;" class="{score_class(tech['score'])}">{tech['score']}</div>
                <div style="font-size:10px; color:#3D5040;">/ 100</div>
            </div>
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:2px; color:#3D5040; text-transform:uppercase;">Fundamental Score</div>
                <div style="font-family:'Syne',sans-serif; font-size:32px; font-weight:700;" class="{score_class(fund['overall'])}">{fund['overall']}</div>
                <div style="font-size:10px; color:#3D5040;">/ 100</div>
            </div>
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:2px; color:#3D5040; text-transform:uppercase;">Technical Bias</div>
                <div style="margin-top:8px;">{verdict_tag(tech['verdict'])}</div>
            </div>
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:2px; color:#3D5040; text-transform:uppercase;">Fundamental Bias</div>
                <div style="margin-top:8px;">{verdict_tag(fund['verdict'])}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 TECHNICAL CHART",
        "🔬 DEEP TECHNICAL",
        "💎 FUNDAMENTALS",
        "🤖 AI ANALYSIS",
        "📋 DATA ROOM"
    ])

    # ════════════════ TAB 1: CHART ═══════════════════════════

    with tab1:
        if not hist_6mo.empty:
            st.plotly_chart(
                chart_candlestick(hist_6mo, indicators, ticker),
                use_container_width=True,
                config={"displayModeBar": False}
            )
        else:
            st.warning("Not enough price history for charting.")

        # Signal Grid
        st.markdown("<div class='section-title'>SIGNAL MATRIX</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (sig_name, sig_type) in enumerate(tech["signals"]):
            tag_cls = tag_for_signal(sig_type)
            cols[i % 4].markdown(
                f'<div style="margin-bottom:8px;"><span class="signal-tag {tag_cls}">{sig_type}</span><br>'
                f'<span style="font-size:10px; color:#3D6040; letter-spacing:1px;">{sig_name}</span></div>',
                unsafe_allow_html=True
            )

    # ════════════════ TAB 2: DEEP TECHNICAL ══════════════════

    with tab2:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>KEY LEVELS</div>", unsafe_allow_html=True)

            levels = [
                ("Resistance 2", tech["r2"], "#FF4D6D"),
                ("Resistance 1", tech["r1"], "#FF8C50"),
                ("Pivot", tech["pivot"], "#F5C842"),
                ("VWAP", tech["vwap"], "#00BFFF"),
                ("Current Price", tech["price"], "#E0EDE2"),
                ("Support 1", tech["s1"], "#00E87A"),
                ("Support 2", tech["s2"], "#00A855"),
                ("Stop Loss", tech["stop"], "#FF4D6D"),
            ]
            for lbl, val, clr in levels:
                pct = ((val - tech["price"]) / tech["price"]) * 100
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding: 8px 0; border-bottom: 1px solid rgba(0,232,122,0.05);">
                    <span style="font-size:11px; color:#3D6040; letter-spacing:1px;">{lbl}</span>
                    <span style="font-family:'Syne',sans-serif; font-size:16px; font-weight:700; color:{clr};">{val:,.2f}</span>
                    <span style="font-size:10px; color:#2A3D2D;">{pct:+.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>TRADE SETUP</div>", unsafe_allow_html=True)

            for lbl, val, clr in [
                ("Entry Zone", f"Market / ~{tech['price']:,.2f}", "#00BFFF"),
                ("Stop Loss", f"{tech['stop']:,.2f}  ({((tech['stop']-tech['price'])/tech['price']*100):+.1f}%)", "#FF4D6D"),
                ("Target 1", f"{tech['tgt1']:,.2f}  ({((tech['tgt1']-tech['price'])/tech['price']*100):+.1f}%)", "#00E87A"),
                ("Target 2", f"{tech['tgt2']:,.2f}  ({((tech['tgt2']-tech['price'])/tech['price']*100):+.1f}%)", "#00E87A"),
                ("Target 3", f"{tech['tgt3']:,.2f}  ({((tech['tgt3']-tech['price'])/tech['price']*100):+.1f}%)", "#00A855"),
                ("Risk:Reward", f"1 : {tech['rr']}", "#F5C842"),
            ]:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding: 10px 0; border-bottom: 1px solid rgba(0,232,122,0.05);">
                    <span style="font-size:11px; color:#3D6040; letter-spacing:1px;">{lbl}</span>
                    <span style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700; color:{clr};">{val}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Indicator table
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>INDICATOR READINGS</div>", unsafe_allow_html=True)
        ind_data = {
            "RSI (14)": f"{tech['rsi']:.1f}",
            "MACD": f"{tech['macd']:.3f}",
            "Signal": f"{tech['macd_signal']:.3f}",
            "ADX": f"{tech['adx']:.1f}",
            "ATR": f"{tech['atr']:.2f}",
            "SMA 20": f"{tech['sma20']:.2f}",
            "SMA 50": f"{tech['sma50']:.2f}",
            "SMA 200": f"{tech['sma200']:.2f}",
            "VWAP": f"{tech['vwap']:.2f}",
            "BB Position": f"{tech['bb_pos']*100:.0f}th %ile",
        }
        ic1, ic2, ic3, ic4, ic5 = st.columns(5)
        ind_cols = [ic1, ic2, ic3, ic4, ic5]
        for i, (k, v) in enumerate(ind_data.items()):
            ind_cols[i % 5].metric(k, v)
        st.markdown("</div>", unsafe_allow_html=True)

    # ════════════════ TAB 3: FUNDAMENTALS ════════════════════

    with tab3:
        f1, f2 = st.columns([1, 1])

        with f1:
            st.plotly_chart(chart_fundamentals(fund), use_container_width=True, config={"displayModeBar": False})

        with f2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>CATEGORY SCORES</div>", unsafe_allow_html=True)
            for cat, score in fund["scores"].items():
                clr = "#00E87A" if score >= 65 else "#F5C842" if score >= 45 else "#FF4D6D"
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                        <span style="font-size:11px; color:#3D6040; letter-spacing:1px;">{cat.upper()}</span>
                        <span style="font-family:'Syne',sans-serif; font-size:14px; font-weight:700; color:{clr};">{score}</span>
                    </div>
                    <div style="background:rgba(0,232,122,0.05); border-radius:4px; height:4px; overflow:hidden;">
                        <div style="background:{clr}; width:{score}%; height:100%; border-radius:4px; opacity:0.8;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Valuation Table
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>VALUATION METRICS</div>", unsafe_allow_html=True)
        vt1, vt2, vt3 = st.columns(3)
        vt1.metric("P/E (TTM)", f"{fund['pe']:.1f}" if fund['pe'] else "N/A")
        vt1.metric("Forward P/E", f"{fund['fwd_pe']:.1f}" if fund['fwd_pe'] else "N/A")
        vt1.metric("PEG Ratio", f"{fund['peg']:.2f}" if fund['peg'] else "N/A")
        vt2.metric("P/B", f"{fund['pb']:.2f}" if fund['pb'] else "N/A")
        vt2.metric("P/S", f"{fund['ps']:.2f}" if fund['ps'] else "N/A")
        vt2.metric("EV/EBITDA", f"{fund['ev_eb']:.1f}" if fund['ev_eb'] else "N/A")
        vt3.metric("Net Margin", f"{fund['npm']:.1f}%")
        vt3.metric("ROE", f"{fund['roe']:.1f}%")
        vt3.metric("D/E Ratio", f"{fund['de']:.2f}" if fund['de'] else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

        # Growth
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>GROWTH & QUALITY</div>", unsafe_allow_html=True)
        gq1, gq2, gq3, gq4 = st.columns(4)
        gq1.metric("Revenue Growth", f"{fund['rev_g']:.1f}%")
        gq2.metric("Earnings Growth", f"{fund['earn_g']:.1f}%")
        gq3.metric("Insider %", f"{fund['insider_pct']:.1f}%")
        gq4.metric("Institutional %", f"{fund['inst_pct']:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    # ════════════════ TAB 4: AI ANALYSIS ═════════════════════

    with tab4:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px;
                    color:#3D5040; margin-bottom:20px; text-transform:uppercase;">
            ⟡ Powered by Gemini · Expert-level institutional analysis
        </div>
        """, unsafe_allow_html=True)

        ai_col1, ai_col2 = st.columns(2)

        with ai_col1:
            with st.spinner("🤖 Running Deep Fundamental AI Analysis..."):
                fund_ai = run_gemini(build_deep_fundamental_prompt(ticker, info, fund))
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>💎 FUNDAMENTAL AI ANALYSIS</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-body'>{fund_ai}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ai_col2:
            with st.spinner("📈 Running Deep Technical AI Analysis..."):
                df_for_tech = hist_6mo if not hist_6mo.empty else hist_1y
                tech_ai = run_gemini(build_deep_technical_prompt(ticker, info, tech, df_for_tech))
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📈 TECHNICAL AI ANALYSIS</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-body'>{tech_ai}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ════════════════ TAB 5: DATA ROOM ═══════════════════════

    with tab5:
        dr1, dr2 = st.columns(2)

        with dr1:
            st.markdown("<div class='section-title'>COMPANY PROFILE</div>", unsafe_allow_html=True)
            profile_fields = {
                "Full Name": info.get("longName", "N/A"),
                "Exchange": info.get("exchange", "N/A"),
                "Currency": info.get("financialCurrency", "N/A"),
                "Country": info.get("country", "N/A"),
                "Employees": f"{info.get('fullTimeEmployees', 0):,}" if info.get("fullTimeEmployees") else "N/A",
                "Website": info.get("website", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
            }
            for k, v in profile_fields.items():
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between;
                            padding:8px 0; border-bottom:1px solid rgba(0,232,122,0.05);">
                    <span style="font-size:11px; color:#3D5040; letter-spacing:1px;">{k}</span>
                    <span style="font-size:12px; color:#A8C4AC;">{v}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            desc = info.get("longBusinessSummary", "No description available.")
            with st.expander("📄 Business Description"):
                st.markdown(f"<div style='font-size:12px; color:#A8C4AC; line-height:1.8;'>{desc}</div>", unsafe_allow_html=True)

        with dr2:
            # Institutional Holders
            inst_holders, major_holders = fetch_holders(ticker)
            if inst_holders is not None and not inst_holders.empty:
                st.markdown("<div class='section-title'>INSTITUTIONAL HOLDERS</div>", unsafe_allow_html=True)
                st.dataframe(
                    inst_holders.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

            # Earnings
            earnings_dates = fetch_earnings(ticker)
            if earnings_dates is not None and not earnings_dates.empty:
                st.markdown("<div class='section-title'>EARNINGS CALENDAR</div>", unsafe_allow_html=True)
                st.dataframe(
                    earnings_dates.head(8),
                    use_container_width=True
                )

        # Raw financials
        with st.expander("📊 Income Statement (Annual)"):
            if not fin["income"].empty:
                st.dataframe(fin["income"] / 1e7, use_container_width=True)
                st.caption("Values in Crores (÷ 10M)")

        with st.expander("🏦 Balance Sheet (Annual)"):
            if not fin["balance"].empty:
                st.dataframe(fin["balance"] / 1e7, use_container_width=True)
                st.caption("Values in Crores (÷ 10M)")

        with st.expander("💵 Cash Flow Statement"):
            if not fin["cashflow"].empty:
                st.dataframe(fin["cashflow"] / 1e7, use_container_width=True)
                st.caption("Values in Crores (÷ 10M)")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:16px 0 8px;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:3px;
                color:#1F2D22; text-transform:uppercase;">
        ⚠ FINTELLIGENCE generates AI-assisted analysis for educational purposes only.
        Not financial advice. Always do your own research.
        Past performance does not guarantee future results.
    </div>
</div>
""", unsafe_allow_html=True)