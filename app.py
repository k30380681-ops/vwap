import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="VWAP Chart Lab", layout="wide")

st.title("VWAP Chart Lab")
st.caption("원하는 종목(기본 TSLA)에 VWAP / Anchored VWAP을 대입해 확인하는 간단 웹앱")

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("설정")

    ticker = st.text_input("티커", value="TSLA").strip().upper()

    interval = st.selectbox(
        "봉 간격(interval)",
        options=["1d", "1h", "30m", "15m", "5m", "1m"],
        index=0,
        help="분봉(1m/5m/15m/30m)은 야후 데이터 제약으로 기간이 짧을 수 있어요."
    )

    # 권장 기본 기간 설정
    if interval in ["1m", "5m", "15m", "30m", "1h"]:
        default_days = 7 if interval in ["1m"] else 30
        max_hint = "⚠️ 분봉은 조회 가능 기간이 제한될 수 있어요(야후 정책)."
    else:
        default_days = 365
        max_hint = ""

    days = st.number_input("조회 기간(일)", min_value=1, max_value=3650, value=default_days, step=1)
    st.caption(max_hint)

    show_vwap = st.checkbox("VWAP 표시", value=True)

    show_anchored = st.checkbox("Anchored VWAP 표시", value=False)
    anchor_date = st.date_input(
        "Anchored 시작일",
        value=(datetime.now().date() - timedelta(days=min(30, int(days)))),
        help="이 날짜부터 누적 VWAP을 다시 계산해 라인을 그립니다.",
        disabled=not show_anchored
    )

    price_source = st.selectbox(
        "VWAP 계산 가격",
        options=["Typical Price(H+L+C)/3", "Close"],
        index=0,
        help="보통 VWAP은 Typical Price 기반이 많이 쓰입니다."
    )

    theme = st.selectbox("테마", options=["plotly_dark", "plotly_white"], index=0)

# -----------------------------
# Data Fetch
# -----------------------------
@st.cache_data(ttl=60)
def load_data(ticker: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

end_dt = datetime.now()
start_dt = end_dt - timedelta(days=int(days))

df = load_data(ticker, start_dt, end_dt, interval)

if df.empty:
    st.error("데이터를 불러오지 못했어요. 티커/기간/간격을 바꿔서 다시 시도해줘!")
    st.stop()

# -----------------------------
# VWAP 계산 (표준: 누적 (가격*거래량)/누적 거래량)
# -----------------------------
def calc_vwap(data: pd.DataFrame, use_typical: bool = True) -> pd.Series:
    if use_typical:
        price = (data["High"] + data["Low"] + data["Close"]) / 3.0
    else:
        price = data["Close"]

    vol = data["Volume"].astype(float)
    pv = price * vol

    cum_pv = pv.cumsum()
    cum_vol = vol.cumsum().replace(0, np.nan)

    return cum_pv / cum_vol

use_typical = price_source.startswith("Typical")

# 기본 VWAP
if show_vwap:
    df["VWAP"] = calc_vwap(df, use_typical=use_typical)

# Anchored VWAP
if show_anchored:
    anchor_ts = pd.Timestamp(anchor_date)
    df["AVWAP"] = np.nan
    anchored = df[df.index >= anchor_ts].copy()
    if not anchored.empty:
        anchored["AVWAP"] = calc_vwap(anchored, use_typical=use_typical)
        df.loc[anchored.index, "AVWAP"] = anchored["AVWAP"]

# -----------------------------
# Plot (Candlestick + Lines)
# -----------------------------
fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    )
)

if show_vwap and "VWAP" in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VWAP"],
            mode="lines",
            name="VWAP",
        )
    )

if show_anchored and "AVWAP" in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["AVWAP"],
            mode="lines",
            name="Anchored VWAP",
        )
    )

fig.update_layout(
    template=theme,
    height=700,
    xaxis_title="Time",
    yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)

fig.update_xaxes(rangeslider_visible=False)

col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("요약")
    st.write(f"티커: **{ticker}**")
    st.write(f"봉 간격: **{interval}**")
    st.write(f"기간: **{start_dt.date()} ~ {end_dt.date()}**")
    st.write(f"VWAP 계산: **{price_source}**")

    last_close = float(df["Close"].iloc[-1])
    st.metric("마지막 종가", f"{last_close:,.2f}")

    if show_vwap and "VWAP" in df.columns and not pd.isna(df["VWAP"].iloc[-1]):
        last_vwap = float(df["VWAP"].iloc[-1])
        st.metric("마지막 VWAP", f"{last_vwap:,.2f}", delta=f"{(last_close - last_vwap):,.2f}")

    if show_anchored and "AVWAP" in df.columns and not pd.isna(df["AVWAP"].iloc[-1]):
        last_avwap = float(df["AVWAP"].iloc[-1])
        st.metric("마지막 Anchored VWAP", f"{last_avwap:,.2f}", delta=f"{(last_close - last_avwap):,.2f}")

st.divider()
st.subheader("데이터 미리보기")
st.dataframe(df.tail(30))
