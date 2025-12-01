import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from decimal import Decimal, getcontext
import math
import time

getcontext().prec = 28

# CONFIG
MAX_WORKERS = 10
FETCH_LIMIT_DEFAULT = 200
FUTURE_TIMEOUT = 40
CCXT_TIMEOUT_MS = 30_000
TOP_N = 200

TFS = {
    '15m': {'minutes': 15, 'bars_needed': 96},
    '1h':  {'minutes': 60, 'bars_needed': 48},
    '4h':  {'minutes': 240,'bars_needed': 24}
}
TF_WEIGHTS = {'15m': 0.50, '1h': 0.35, '4h': 0.15}
HOURS_PER_YEAR = 24.0 * 365.0

st.set_page_config(page_title="Quant Crypto Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>.block-container {padding-top: 1rem; padding-bottom: 3rem;} div[data-testid="stMetricValue"] {font-size: 1.3rem;}</style>""", unsafe_allow_html=True)

# ---------- NEW: label distance threshold control ----------
label_distance_threshold = st.sidebar.slider(
    "Label distance threshold (fraction of axis range)",
    min_value=0.0,
    max_value=0.20,
    value=0.05,
    step=0.005,
    help="When points are closer than this (normalized fraction of axis range), their labels are hidden to avoid overlap."
)
# ----------------------------------------------------------

# Helpers
def parse_decimal_safe(x):
    if x is None: return 0.0
    try: return float(x)
    except Exception:
        try: return float(str(x).replace(',', ''))
        except Exception: return 0.0

def normalize_funding_rate(raw):
    if raw is None: return 0.0
    try:
        s = str(raw).strip()
        if s.endswith('%'):
            return float(s[:-1].strip()) / 100.0
        v = float(s)
        if abs(v) > 1: return v / 100.0
        return v
    except Exception:
        return 0.0

def robust_z_score(series: pd.Series):
    s = pd.to_numeric(series, errors='coerce').copy()
    valid = s.dropna()
    if valid.empty: return pd.Series(0, index=s.index)
    med = valid.median()
    mad = (valid - med).abs().median()
    if mad == 0 or np.isnan(mad):
        mean = valid.mean()
        std = valid.std(ddof=1)
        if std == 0 or np.isnan(std): return pd.Series(0, index=s.index)
        return (s - mean) / std
    return (s - med) / (1.4826 * mad)

# Per-symbol multi-TF fetcher
def fetch_symbol_timeframes(exchange_id, symbol, tfs, limit_default=FETCH_LIMIT_DEFAULT):
    try:
        exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True, 'timeout': CCXT_TIMEOUT_MS})
        results = {}
        for tf in tfs:
            limit = min(limit_default, max(1, TFS.get(tf, {}).get('bars_needed', limit_default)))
            try:
                results[tf] = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            except Exception:
                results[tf] = None
        try: exchange.close()
        except: pass
        return symbol, results
    except Exception:
        return symbol, None

# Fetch market data (cached)
@st.cache_data(ttl=60, show_spinner=False)
def get_market_data(top_n=TOP_N):
    exchange = ccxt.bybit({'enableRateLimit': True, 'timeout': CCXT_TIMEOUT_MS})
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        try: exchange.close()
        except: pass
        print("fetch_tickers failed:", type(e).__name__, str(e)[:200])
        return pd.DataFrame()

    rows = []
    for s, t in tickers.items():
        if not isinstance(s, str): continue
        if not s.endswith(':USDT'): continue
        qv = t.get('quoteVolume')
        if qv in (None, 0): continue
        info = t.get('info') or {}
        oi = t.get('openInterest') or (info.get('openInterest') if isinstance(info, dict) else None) or (info.get('openInterestValue') if isinstance(info, dict) else None)
        fr = t.get('fundingRate') or (info.get('predictedFunding') if isinstance(info, dict) else None) or (info.get('fundingRate') if isinstance(info, dict) else None)
        rows.append({'symbol': s, 'last_price_snapshot': parse_decimal_safe(t.get('last')), 'quoteVolume': parse_decimal_safe(qv), 'funding_rate_raw': fr, 'open_interest_raw': oi})

    try: exchange.close()
    except: pass

    if not rows: return pd.DataFrame()
    df_tickers = pd.DataFrame(rows).sort_values('quoteVolume', ascending=False).head(top_n)
    symbols = df_tickers['symbol'].tolist()

    tfs = list(TFS.keys())
    history = {}
    workers = min(MAX_WORKERS, max(1, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_symbol_timeframes, 'bybit', s, tfs): s for s in symbols}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                sym, tfdata = fut.result(timeout=FUTURE_TIMEOUT)
                if tfdata: history[sym] = tfdata
            except TimeoutError:
                print(f"[timeout] {s}")
            except Exception as e:
                print(f"[worker error] {s}: {type(e).__name__} {str(e)[:200]}")

    # ---------------- compute raw signals for each symbol that has history ----------------
    final_rows = []
    for _, row in df_tickers.iterrows():
        symbol = row['symbol']
        if symbol not in history: continue
        hist = history[symbol]
        data_15m = hist.get('15m') if hist else None
        data_1h = hist.get('1h') if hist else None
        if not data_15m or not data_1h: continue
        if len(data_15m) < 30 or len(data_1h) < 30: continue

        # local helper to compute the signals at the last bar (exactly as you posted)
        def calc_signals(ohlcv):
            df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
            c = df['c'].astype(float)
            h = df['h'].astype(float)
            l = df['l'].astype(float)
            o_ = df['o'].astype(float)
            window = 20
            sma = c.rolling(window).mean()
            std = c.rolling(window).std()
            cp = c.shift()
            tr1 = h - l
            tr2 = (h - cp).abs()
            tr3 = (l - cp).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window).mean()
            ema = c.ewm(span=window, adjust=False).mean()
            roll_max = h.rolling(window).max()
            roll_min = l.rolling(window).min()

            # Bollinger + Keltner mix
            bb_score = (c - sma) / (2 * std.replace(0, np.nan))
            kc_score = (c - ema) / (1.5 * atr.replace(0, np.nan))
            sig_1 = (bb_score + kc_score) / 2

            # Donchian position centered
            range_width = (roll_max - roll_min).replace(0, np.nan)
            sig_2 = ((c - roll_min) / range_width) - 0.5

            # Rob Carver style
            sig_3 = (c - sma) / range_width

            # Larry Williams / %R style (close-open)/atr units
            sig_4 = (c - o_) / atr.replace(0, np.nan)

            # pick last non-nan values, fallback to 0
            def last_or_zero(series):
                try:
                    v = series.iloc[-1]
                    if np.isfinite(v):
                        return float(v)
                    else:
                        return 0.0
                except Exception:
                    return 0.0

            return {
                's1': last_or_zero(sig_1),
                's2': last_or_zero(sig_2),
                's3': last_or_zero(sig_3),
                's4': last_or_zero(sig_4),
                'close': float(c.iloc[-1])
            }

        sigs_15 = calc_signals(data_15m)
        sigs_1h = calc_signals(data_1h)

        # volatility simple: sample std of 1h log-returns annualized (this is your original formula)
        df_vol = pd.DataFrame(data_1h, columns=['ts','o','h','l','c','v'])
        df_vol['c'] = pd.to_numeric(df_vol['c'], errors='coerce')
        returns = np.log(df_vol['c'] / df_vol['c'].shift(1)).dropna()
        volatility_simple = 0.0
        if len(returns) >= 2:
            volatility_simple = float(returns.std(ddof=1, skipna=True) * np.sqrt(24 * 365))
        else:
            volatility_simple = 0.0

        # ALSO keep the corrected EWMA volatility (sqrt(EWMA_var) * sqrt(periods per year))
        vol_annual = 0.0
        lr = returns  # same series
        if len(lr) >= 2:
            ewma_var = lr.ewm(span=24, adjust=False).var().iloc[-1]
            if np.isfinite(ewma_var) and ewma_var >= 0:
                vol_annual = float(np.sqrt(ewma_var) * np.sqrt(HOURS_PER_YEAR))
            else:
                sample_std = float(lr.tail(24).std(ddof=1)) if len(lr.tail(24)) >= 2 else 0.0
                vol_annual = float(sample_std * np.sqrt(HOURS_PER_YEAR)) if sample_std and np.isfinite(sample_std) else 0.0
        else:
            vol_annual = 0.0

        # price and pct change (1d)
        close_now = sigs_1h['close']
        close_24h = df_vol['c'].iloc[-25] if len(df_vol) >= 25 else df_vol['c'].iloc[0]
        pct_change = (close_now - close_24h) / close_24h if close_24h != 0 else 0.0

        # restore vol_breakout (MA24 / STD24)
        last24 = df_vol['c'].tail(24).astype(float)
        if len(last24) >= 2:
            ma_24 = last24.mean()
            std_24 = last24.std(ddof=1)
            vol_breakout = (close_now - ma_24) / std_24 if std_24 > 0 else 0.0
        else:
            vol_breakout = 0.0

        # funding / carry (as before)
        fr = normalize_funding_rate(row.get('funding_rate_raw'))
        funding_apr = fr * 3 * 365 * 100

        final_rows.append({
            'symbol': symbol.split(':')[0],
            'price': close_now,
            'pct_change_1d': pct_change * 100,
            'pct_change_7d': 0.0,   # keep compatibility (you can compute from longer history if desired)
            'volatility': vol_annual,                # EWMA-based annualized volatility (decimal)
            'volatility_simple': volatility_simple,   # your original std-based annualized volatility (decimal)
            'open_interest': parse_decimal_safe(row.get('open_interest_raw')),
            'funding_apr': funding_apr,
            'donchian_score_1h': (sigs_1h['s2'] if 's2' in sigs_1h else 0.0),  # sample mapping if you want to inspect
            'donchian_score_15m': (sigs_15['s2'] if 's2' in sigs_15 else 0.0),
            'donchian_score': sigs_1h['s2'],
            'vol_breakout': vol_breakout,
            's1_15': sigs_15['s1'], 's2_15': sigs_15['s2'], 's3_15': sigs_15['s3'], 's4_15': sigs_15['s4'],
            's1_1h': sigs_1h['s1'], 's2_1h': sigs_1h['s2'], 's3_1h': sigs_1h['s3'], 's4_1h': sigs_1h['s4'],
            # store returns arrays for CR-Momo later (full lr arrays)
            'ret_15m': np.log(pd.DataFrame(data_15m, columns=['ts','o','h','l','c','v'])['c'].astype(float).values[1:] / pd.DataFrame(data_15m, columns=['ts','o','h','l','c','v'])['c'].astype(float).values[:-1]) if len(data_15m) >= 2 else None,
            'ret_1h': returns.values if len(returns) >= 1 else None,
            'ret_4h': (np.log(pd.DataFrame(hist.get('4h', []), columns=['ts','o','h','l','c','v'])['c'].astype(float).values[1:] / pd.DataFrame(hist.get('4h', []), columns=['ts','o','h','l','c','v'])['c'].astype(float).values[:-1]) if hist.get('4h') and len(hist.get('4h')) >= 2 else None)
        })

    df_out = pd.DataFrame(final_rows)
    return df_out

# Processing and CR-Momo (same robust alignment & combination as before)
def process_data(df):
    if df is None or df.empty: return df
    df = df.copy()
    # compute breakout subsignals z-scores
    for col in ['s1_15','s2_15','s3_15','s4_15','s1_1h','s2_1h','s3_1h','s4_1h']:
        if col in df.columns: df[f'z_{col}'] = robust_z_score(df[col])
        else: df[f'z_{col}'] = 0.0
    df['vol_breakout_score'] = df[[f'z_{c}' for c in ['s1_15','s2_15','s3_15','s4_15','s1_1h','s2_1h','s3_1h','s4_1h']]].mean(axis=1)
    df['z_price'] = robust_z_score(df.get('pct_change_1d', pd.Series(0, index=df.index)))
    df['z_vol'] = robust_z_score(df.get('volatility', pd.Series(0, index=df.index)))
    df['z_oi'] = robust_z_score(np.log1p(df.get('open_interest', pd.Series(0, index=df.index))))

    # returns collection per TF
    tf_residuals = {tf: {} for tf in TFS.keys()}
    symbols = df['symbol'].tolist()
    for _, row in df.iterrows():
        sym = row['symbol']
        for tf in TFS.keys():
            arr = row.get(f'ret_{tf}')
            if arr is None: continue
            try:
                a = np.array(arr, dtype=float)
                if a.size >= 1: tf_residuals[tf][sym] = a
            except Exception:
                continue

    # compute residual zscore per TF
    z_by_tf = {}
    for tf, mapping in tf_residuals.items():
        syms = list(mapping.keys())
        if len(syms) == 0:
            z_by_tf[tf] = pd.Series(np.nan, index=symbols); continue
        lengths = [mapping[s].size for s in syms]
        L = min(lengths)
        if L < 1:
            z_by_tf[tf] = pd.Series(np.nan, index=symbols); continue
        M = np.vstack([mapping[s][-L:] for s in syms])
        market = np.nanmedian(M, axis=0)
        m_mean = np.nanmean(market)
        m_var = np.nanvar(market, ddof=1) if L > 1 else 0.0
        resid_map = {}
        for i, s in enumerate(syms):
            a = M[i, :]
            a_mean = np.nanmean(a)
            cov = np.nanmean((a - a_mean) * (market - m_mean))
            beta = cov / m_var if (m_var and not np.isnan(m_var)) else 0.0
            resid_last = a[-1] - beta * market[-1]
            resid_pct = resid_last * 100.0
            resid_map[s] = resid_pct
        resid_series = pd.Series([resid_map.get(s, np.nan) for s in symbols], index=symbols)
        z_by_tf[tf] = robust_z_score(resid_series)

    cr_raw = pd.Series(0.0, index=symbols, dtype=float)
    total_weight = 0.0
    for tf, weight in TF_WEIGHTS.items():
        if tf in z_by_tf:
            cr_raw += z_by_tf[tf].fillna(0.0) * float(weight)
            total_weight += float(weight)
    if total_weight == 0: cr_raw = pd.Series(0.0, index=symbols)
    else: cr_raw = cr_raw / total_weight

    cr_final = robust_z_score(cr_raw)
    df['cr_momo_raw'] = [cr_raw.get(s, np.nan) for s in symbols]
    df['cr_momo_z'] = [cr_final.get(s, np.nan) for s in symbols]
    df['resmom'] = df['cr_momo_z'].fillna(0.0)
    df['comp_momo'] = (df['z_price'].fillna(0.0) + df['resmom'].fillna(0.0)) / 2.0

    return df

# Plot helpers and UI (unchanged; include volatility_simple and vol_breakout in debug)
def make_scatter(df, x, y, color_col, title, xl, yl, label_threshold=None):
    # compute symbols to display as text depending on nearest neighbour distance (normalized)
    xs = pd.to_numeric(df[x], errors='coerce').fillna(0).values
    ys = pd.to_numeric(df[y], errors='coerce').fillna(0).values
    symbols = df['symbol'].tolist()

    # default: show all symbols if no threshold supplied
    if label_threshold is None:
        texts = symbols
    else:
        # normalize by axis ranges to make threshold fraction meaningful
        xmin, xmax = np.nanmin(xs), np.nanmax(xs)
        ymin, ymax = np.nanmin(ys), np.nanmax(ys)
        xrange = (xmax - xmin) if (xmax > xmin) else 1.0
        yrange = (ymax - ymin) if (ymax > ymin) else 1.0
        nx = (xs - xmin) / xrange
        ny = (ys - ymin) / yrange
        pts = np.vstack([nx, ny]).T

        # pairwise distances (OK for TOP_N ~ 200)
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        # ignore self-distance
        np.fill_diagonal(d, np.inf)
        min_d = d.min(axis=1)

        texts = [s if md >= label_threshold else "" for s, md in zip(symbols, min_d)]

    fig = px.scatter(df, x=x, y=y, color=color_col, text=texts, title=f"<b>{title}</b>", color_continuous_scale='RdBu', hover_data=['price','pct_change_1d','open_interest','funding_apr','volatility','volatility_simple','vol_breakout'])
    fig.update_traces(textposition='top center', marker=dict(size=10, line=dict(width=1, color='#333')))
    fig.update_layout(height=450, template="plotly_white", margin=dict(t=50,b=20,l=20,r=20), xaxis_title=xl, yaxis_title=yl)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
    return fig

def make_bar(df, x, title):
    df_sorted = df.sort_values(x, ascending=True, na_position='first')
    subset = pd.concat([df_sorted.head(10), df_sorted.tail(10)])
    fig = px.bar(subset, x=x, y='symbol', orientation='h', color=x, title=f"<b>{title}</b>", color_continuous_scale='RdBu')
    fig.update_layout(height=600, template="plotly_white", yaxis=dict(autorange=True), coloraxis_showscale=False)
    return fig

st.title("Quant Crypto Dashboard)")

if st.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()

with st.spinner("Fetching multi-TF data and computing CR-Momo..."):
    raw_df = get_market_data()

if raw_df is None or raw_df.empty:
    st.error("No data returned — check connection or reduce TOP_N / MAX_WORKERS.")
else:
    df = process_data(raw_df)
    # metrics
    def safe_select(df, col, agg='max'):
        if df.empty or col not in df.columns: return None
        valid = df[col].dropna()
        if valid.empty: return None
        idx = valid.idxmax() if agg == 'max' else valid.idxmin()
        return df.loc[idx]

    best = safe_select(df, 'pct_change_1d', 'max')
    worst = safe_select(df, 'pct_change_1d', 'min')
    carry_king = safe_select(df, 'funding_apr', 'max')
    m1, m2, m3 = st.columns(3)
    if best is not None: m1.metric("Top Gainer (24h)", best['symbol'], f"{best['pct_change_1d']:.2f}%")
    else: m1.metric("Top Gainer (24h)","-","-")
    if worst is not None: m2.metric("Top Loser (24h)", worst['symbol'], f"{worst['pct_change_1d']:.2f}%")
    else: m2.metric("Top Loser (24h)","-","-")
    if 'volatility' in df.columns and df['volatility'].notna().any():
        vol_med = float(df['volatility'].median(skipna=True))
        m3.metric("Market Volatility (median)", f"{vol_med*100:.1f}%", "Annualized")
    else:
        m3.metric("Market Volatility (median)", "-", "Annualized")

    st.divider()
    tab1, tab2 = st.tabs(["Breakout","Momentum & Carry"])
    with tab1:
        c1, c2 = st.columns(2)
        if 'z_oi' in df.columns and 'z_price' in df.columns:
            # pass the user-controlled threshold here
            c1.plotly_chart(make_scatter(df, 'z_oi', 'z_price', 'pct_change_1d', "OI vs Price (Z-Scores)","OI Z-Score","Price Z-Score", label_threshold=label_distance_threshold), width='stretch')
        c2.plotly_chart(make_scatter(df, 'volatility_simple', 'pct_change_1d', 'pct_change_1d', "Volatility (simple) vs Returns","Volatility (simple decimal)","Returns 24h (%)", label_threshold=label_distance_threshold), width='stretch')
        try: st.plotly_chart(make_bar(df, 'vol_breakout_score', "Composite Breakout Score"), width='stretch')
        except Exception as e: st.write("Bar plot failed:", e)
    with tab2:
        c3, c4 = st.columns(2)
        c3.plotly_chart(make_scatter(df, 'resmom', 'pct_change_1d', 'funding_apr', "CR-Residual vs Returns","CR-Residual (z)","Returns 24h (%)", label_threshold=label_distance_threshold), width='stretch')
        c4.plotly_chart(make_scatter(df, 'funding_apr', 'z_price', 'funding_apr', "Carry vs Price Strength","Funding APR","Price Z-Score", label_threshold=label_distance_threshold), width='stretch')
        try: st.plotly_chart(make_bar(df, 'comp_momo', "Composite Momentum (price + CR)"), width='stretch')
        except Exception: pass

    if st.checkbox("Show debug table (first 200 rows)", False):
        show_cols = [c for c in ['symbol','pct_change_1d','volatility','volatility_simple','volatility_pct','funding_apr','vol_breakout','cr_momo_raw','cr_momo_z','comp_momo'] if c in df.columns]
        st.dataframe(df[show_cols].head(200))
