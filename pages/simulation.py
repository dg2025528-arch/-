
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ── 페이지 설정 ───────────────────────────────────────────────────
st.set_page_config(
    page_title="미래 해양 생태계 시뮬레이션",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 공통 색상 ─────────────────────────────────────────────────────
PLOT_BG    = "#0a1a2e"
PAPER_BG   = "#0d2137"
FONT_COLOR = "#c8eef5"
ACCENT     = "#00d4ff"

def style_fig(fig, title=None, height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(color=ACCENT, size=16)) if title else None,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(13,33,55,0.6)", bordercolor=ACCENT, borderwidth=1),
        hoverlabel=dict(bgcolor="#0d2137", font_size=12),
    )
    fig.update_xaxes(gridcolor="rgba(0,212,255,0.1)",
                     zerolinecolor="rgba(0,212,255,0.2)", color=FONT_COLOR)
    fig.update_yaxes(gridcolor="rgba(0,212,255,0.1)",
                     zerolinecolor="rgba(0,212,255,0.2)", color=FONT_COLOR)
    return fig

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
    }
    .main-title {
        text-align: center; font-size: 2.4rem; font-weight: 900;
        color: #00d4ff; text-shadow: 0 0 30px rgba(0,212,255,0.5);
        padding: 20px 0 5px 0; letter-spacing: 2px;
    }
    .sub-title {
        text-align: center; font-size: 1.0rem;
        color: #7ecfdf; padding-bottom: 20px;
    }
    .section-header {
        color: #00d4ff; font-size: 1.3rem; font-weight: 700;
        border-left: 4px solid #00d4ff;
        padding-left: 12px; margin: 25px 0 15px 0;
    }
    .info-box {
        background: rgba(0,212,255,0.08); border: 1px solid #00d4ff55;
        border-radius: 10px; padding: 15px 20px;
        color: #c8eef5; font-size: 0.95rem; line-height: 1.7;
    }
    .warning-box {
        background: rgba(255,165,0,0.1); border: 1px solid #ffa50066;
        border-radius: 10px; padding: 15px 20px;
        color: #ffd580; font-size: 0.95rem; line-height: 1.7;
    }
    .result-card {
        border-radius: 14px; padding: 25px;
        text-align: center; font-weight: bold;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .result-safe {
        background: linear-gradient(135deg, #0d3e1f, #0a4d26);
        border: 2px solid #2ecc71;
    }
    .result-warning {
        background: linear-gradient(135deg, #3e2d0a, #4d380a);
        border: 2px solid #f39c12;
    }
    .result-danger {
        background: linear-gradient(135deg, #3e0d0d, #4d1010);
        border: 2px solid #e74c3c;
    }
    .result-value {
        font-size: 3rem; font-weight: 900; margin: 10px 0;
    }
    .result-label {
        font-size: 0.9rem; color: #c8eef5; margin-top: 5px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1e30, #0d2a3e);
        border-right: 1px solid #00d4ff33;
    }
    .stMarkdown, p, span, div { color: #c8eef5; }
    h1, h2, h3 { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ── 데이터 로드 ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("realistic_ocean_climate_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    severity_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    df["Bleaching_Num"] = df["Bleaching Severity"].map(severity_map)
    df["Heatwave_Num"] = df["Marine Heatwave"].map(
        {True: 1, False: 0, "TRUE": 1, "FALSE": 0}
    )
    return df

df = load_data()

# ── 회귀모델 학습 ─────────────────────────────────────────────────
@st.cache_data
def train_model(df):
    features = ["pH Level", "SST (°C)", "Bleaching_Num", "Heatwave_Num"]
    df_model = df.dropna(subset=features + ["Species Observed"])
    X = df_model[features].values
    y = df_model["Species Observed"].values
    model = LinearRegression()
    model.fit(X, y)
    score = model.score(X, y)
    return model, score

# 연도별 추세 회귀
@st.cache_data
def calc_yearly_trend(df):
    yearly = df.groupby("Year").agg(
        pH_mean=("pH Level", "mean"),
        SST_mean=("SST (°C)", "mean"),
        Species_mean=("Species Observed", "mean"),
    ).reset_index()
    trends = {}
    for col in ["pH_mean", "SST_mean", "Species_mean"]:
        slope, intercept, r, p, _ = stats.linregress(yearly["Year"], yearly[col])
        trends[col] = {"slope": slope, "intercept": intercept, "r": r}
    return yearly, trends

model, model_score = train_model(df)
yearly, trends = calc_yearly_trend(df)

# ── 헤더 ─────────────────────────────────────────────────────────
st.markdown("""
<div class='main-title'>🔮 미래 해양 생태계 시뮬레이션</div>
<div class='sub-title'>Future Ocean Ecosystem Simulation</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
🔬 이 페이지는 <b>2015~2023년 실제 관측 데이터</b>를 기반으로 학습한
회귀 모델을 사용합니다. <b>PART 1</b>에서 현재 추세대로 갔을 때의 미래를 보고,
<b>PART 2</b>에서 직접 환경 조건을 바꿔 종 수 변화를 시뮬레이션해보세요.<br>
⚠️ 이 예측은 <b>탐구 목적의 추정값</b>이며, 실제 미래와 다를 수 있습니다.
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PART 1. 미래 추세 예측 (2024~2030)
# ════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📈 PART 1. 현재 추세대로라면? (2024~2030 예측)</div>",
            unsafe_allow_html=True)
st.markdown("""
<div class='warning-box'>
📊 2015~2023년 데이터의 <b>추세선을 2030년까지 연장</b>한 예측입니다.
실선은 실제 관측값, <b>점선은 예측값</b>이에요.
</div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# 예측 연도
future_years = list(range(2024, 2031))
all_years    = list(yearly["Year"]) + future_years

# 예측값 계산
pred_ph  = [trends["pH_mean"]["slope"] * y + trends["pH_mean"]["intercept"]
            for y in future_years]
pred_sst = [trends["SST_mean"]["slope"] * y + trends["SST_mean"]["intercept"]
            for y in future_years]
pred_sp  = [trends["Species_mean"]["slope"] * y + trends["Species_mean"]["intercept"]
            for y in future_years]

# 3개 서브플롯
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("평균 pH 예측", "평균 수온(°C) 예측", "평균 관측 종 수 예측")
)

plot_data = [
    (1, "pH_mean", pred_ph, "#00d4ff", "pH"),
    (2, "SST_mean", pred_sst, "#ff6b6b", "수온(°C)"),
    (3, "Species_mean", pred_sp, "#2ecc71", "종 수"),
]

for col_n, obs_col, pred_vals, color, name in plot_data:
    # 실제 관측값 (실선)
    fig.add_trace(go.Scatter(
        x=list(yearly["Year"]), y=list(yearly[obs_col]),
        mode="lines+markers", name=f"실제 {name}",
        line=dict(color=color, width=3), marker=dict(size=7),
        showlegend=(col_n == 1)
    ), row=1, col=col_n)

    # 연결선 (실제 마지막 ~ 예측 첫값)
    fig.add_trace(go.Scatter(
        x=[yearly["Year"].iloc[-1], future_years[0]],
        y=[yearly[obs_col].iloc[-1], pred_vals[0]],
        mode="lines", line=dict(color=color, width=2, dash="dot"),
        showlegend=False
    ), row=1, col=col_n)

    # 예측값 (점선)
    fig.add_trace(go.Scatter(
        x=future_years, y=pred_vals,
        mode="lines+markers", name=f"예측 {name}",
        line=dict(color=color, width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
        showlegend=(col_n == 1)
    ), row=1, col=col_n)

    # 예측 구간 음영
    fig.add_vrect(
        x0=2023.5, x1=2030.5,
        fillcolor=color, opacity=0.05,
        layer="below", line_width=0,
        row=1, col=col_n
    )

fig = style_fig(fig, "현재 추세 연장 예측 (점선: 예측 구간)", height=480)
fig.update_annotations(font=dict(color=ACCENT, size=13))
st.plotly_chart(fig, use_container_width=True)

# 2030년 예측값 카드
st.markdown("##### 📌 2030년 예측값 요약")
c1, c2, c3 = st.columns(3)

def trend_card(col, label, current, predicted, unit, reverse=False):
    diff = predicted - current
    is_bad = (diff < 0) if not reverse else (diff > 0)
    color = "#e74c3c" if is_bad else "#2ecc71"
    arrow = "▼" if diff < 0 else "▲"
    col.markdown(f"""
    <div style='background:{"rgba(231,76,60,0.1)" if is_bad else "rgba(46,204,113,0.1)"};
         border:1px solid {color}; border-radius:12px; padding:18px; text-align:center;'>
        <div style='color:#7ecfdf; font-size:0.85rem;'>{label}</div>
        <div style='color:{color}; font-size:2rem; font-weight:900;'>{predicted:.2f}{unit}</div>
        <div style='color:#c8eef5; font-size:0.85rem;'>현재 평균: {current:.2f}{unit}</div>
        <div style='color:{color}; font-size:1rem; font-weight:bold;'>
            {arrow} {abs(diff):.2f}{unit}
        </div>
    </div>""", unsafe_allow_html=True)

trend_card(c1, "🧪 2030년 예상 pH",
           yearly["pH_mean"].mean(), pred_ph[-1], "", reverse=False)
trend_card(c2, "🌡️ 2030년 예상 수온",
           yearly["SST_mean"].mean(), pred_sst[-1], "°C", reverse=True)
trend_card(c3, "🐠 2030년 예상 종 수",
           yearly["Species_mean"].mean(), pred_sp[-1], "종", reverse=False)

st.markdown(f"""
<br>
<div class='info-box'>
📌 <b>추세 해석</b><br>
• pH: 연간 <b>{trends['pH_mean']['slope']:+.4f}</b> 변화
  → 2030년까지 {'감소(산성화 진행)' if trends['pH_mean']['slope'] < 0 else '증가'} 예상<br>
• 수온: 연간 <b>{trends['SST_mean']['slope']:+.4f}°C</b> 변화
  → 2030년까지 {'상승' if trends['SST_mean']['slope'] > 0 else '하강'} 예상<br>
• 관측 종 수: 연간 <b>{trends['Species_mean']['slope']:+.4f}종</b> 변화
  → 2030년까지 {'감소' if trends['Species_mean']['slope'] < 0 else '증가'} 예상
</div>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PART 2. 직접 조작 시뮬레이션
# ════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🎚️ PART 2. 내가 직접 환경을 바꿔보면?</div>",
            unsafe_allow_html=True)
st.markdown("""
<div class='warning-box'>
🎮 아래 슬라이더로 해양 환경 조건을 직접 바꿔보세요.
조건을 바꾸면 <b>예상 생물종 수가 실시간으로 변화</b>합니다.<br>
실제 데이터 범위:
