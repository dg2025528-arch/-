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
PLOT_BG = "#0a1a2e"
PAPER_BG = "#0d2137"
FONT_COLOR = "#c8eef5"
ACCENT = "#00d4ff"


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
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    severity_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    df["Bleaching_Num"] = df["Bleaching Severity"].map(severity_map)
    df["Heatwave_Num"] = df["Marine Heatwave"].map(
        {True: 1, False: 0, "TRUE": 1, "FALSE": 0}
    )
    return df


df = load_data()


# ── 회귀모델 학습 ─────────────────────────────────────────────────
@st.cache_resource
def train_model(_df):
    features = ["pH Level", "SST (°C)", "Bleaching_Num", "Heatwave_Num"]
    df_model = _df.dropna(subset=features + ["Species Observed"])
    X = df_model[features].values
    y = df_model["Species Observed"].values
    reg = LinearRegression()
    reg.fit(X, y)
    score = reg.score(X, y)
    return reg, score


@st.cache_data
def calc_yearly_trend(_df):
    yearly_df = _df.groupby("Year").agg(
        pH_mean=("pH Level", "mean"),
        SST_mean=("SST (°C)", "mean"),
        Species_mean=("Species Observed", "mean"),
    ).reset_index()
    trend_dict = {}
    for col in ["pH_mean", "SST_mean", "Species_mean"]:
        slope, intercept, r, p, _ = stats.linregress(yearly_df["Year"], yearly_df[col])
        trend_dict[col] = {"slope": slope, "intercept": intercept, "r": r}
    return yearly_df, trend_dict


model, model_score = train_model(df)
yearly, trends = calc_yearly_trend(df)

# 자주 쓰는 평균값 미리 계산
ph_avg = df["pH Level"].mean()
sst_avg = df["SST (°C)"].mean()
current_avg = df["Species Observed"].mean()

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
# 사이드바 슬라이더 (먼저 정의)
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎚️ 환경 조건 설정")
    st.markdown("---")

    st.markdown("### 🧪 pH 설정")
    sim_ph = st.slider("pH 수준", min_value=7.6, max_value=8.4,
                       value=float(ph_avg), step=0.01,
                       help="낮을수록 산성화 심각")

    st.markdown("### 🌡️ 수온 설정")
    sim_sst = st.slider("수면 수온 (°C)", min_value=22.0, max_value=35.0,
                        value=float(sst_avg), step=0.1,
                        help="높을수록 열파 위험 증가")

    st.markdown("### 🪸 백화 심각도")
    sev_options = {"없음": 0, "낮음": 1, "중간": 2, "높음": 3}
    sim_sev_label = st.select_slider("백화 심각도",
                                     options=["없음", "낮음", "중간", "높음"],
                                     value="낮음")
    sim_sev = sev_options[sim_sev_label]

    st.markdown("### 🌊 해양열파")
    sim_hw = st.radio("해양열파 발생", ["없음", "있음"])
    sim_hw_num = 1 if sim_hw == "있음" else 0

    st.markdown("---")
    st.markdown("### 📊 현재 데이터 평균")
    st.markdown(f"- **평균 pH**: {ph_avg:.3f}")
    st.markdown(f"- **평균 수온**: {sst_avg:.1f}°C")
    st.markdown(f"- **평균 종 수**: {current_avg:.0f}종")


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

future_years = list(range(2024, 2031))

pred_ph = [trends["pH_mean"]["slope"] * yr + trends["pH_mean"]["intercept"]
           for yr in future_years]
pred_sst = [trends["SST_mean"]["slope"] * yr + trends["SST_mean"]["intercept"]
            for yr in future_years]
pred_sp = [trends["Species_mean"]["slope"] * yr + trends["Species_mean"]["intercept"]
           for yr in future_years]

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("평균 pH 예측", "평균 수온(°C) 예측", "평균 관측 종 수 예측")
)

plot_data = [
    (1, "pH_mean", pred_ph, "#00d4ff", "pH"),
    (2, "SST_mean", pred_sst, "#ff6b6b", "수온"),
    (3, "Species_mean", pred_sp, "#2ecc71", "종수"),
]

for col_n, obs_col, pred_vals, color, name in plot_data:
    fig.add_trace(go.Scatter(
        x=list(yearly["Year"]), y=list(yearly[obs_col]),
        mode="lines+markers", name=f"실제 {name}",
        line=dict(color=color, width=3), marker=dict(size=7),
        showlegend=(col_n == 1)
    ), row=1, col=col_n)

    fig.add_trace(go.Scatter(
        x=[int(yearly["Year"].iloc[-1]), future_years[0]],
        y=[float(yearly[obs_col].iloc[-1]), pred_vals[0]],
        mode="lines", line=dict(color=color, width=2, dash="dot"),
        showlegend=False
    ), row=1, col=col_n)

    fig.add_trace(go.Scatter(
        x=future_years, y=pred_vals,
        mode="lines+markers", name=f"예측 {name}",
        line=dict(color=color, width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
        showlegend=(col_n == 1)
    ), row=1, col=col_n)

    fig.add_vrect(
        x0=2023.5, x1=2030.5,
        fillcolor=color, opacity=0.05,
        layer="below", line_width=0,
        row=1, col=col_n
    )

fig = style_fig(fig, "현재 추세 연장 예측 (점선: 예측 구간)", height=480)
fig.update_annotations(font=dict(color=ACCENT, size=13))
st.plotly_chart(fig, use_container_width=True)

st.markdown("##### 📌 2030년 예측값 요약")
c1, c2, c3 = st.columns(3)


def trend_card(col, label, current, predicted, unit, reverse=False):
    diff = predicted - current
    is_bad = (diff < 0) if not reverse else (diff > 0)
    color = "#e74c3c" if is_bad else "#2ecc71"
    arrow = "▼" if diff < 0 else "▲"
    bg = "rgba(231,76,60,0.1)" if is_bad else "rgba(46,204,113,0.1)"
    col.markdown(f"""
    <div style='background:{bg}; border:1px solid {color};
         border-radius:12px; padding:18px; text-align:center;'>
        <div style='color:#7ecfdf; font-size:0.85rem;'>{label}</div>
        <div style='color:{color}; font-size:2rem; font-weight:900;'>{predicted:.2f}{unit}</div>
        <div style='color:#c8eef5; font-size:0.85rem;'>현재 평균: {current:.2f}{unit}</div>
        <div style='color:{color}; font-size:1rem; font-weight:bold;'>{arrow} {abs(diff):.2f}{unit}</div>
    </div>""", unsafe_allow_html=True)


trend_card(c1, "🧪 2030년 예상 pH",
           yearly["pH_mean"].mean(), pred_ph[-1], "", reverse=False)
trend_card(c2, "🌡️ 2030년 예상 수온",
           yearly["SST_mean"].mean(), pred_sst[-1], "°C", reverse=True)
trend_card(c3, "🐠 2030년 예상 종 수",
           yearly["Species_mean"].mean(), pred_sp[-1], "종", reverse=False)

# f-string 밖에서 미리 해석 문장 계산
ph_trend_txt = "감소(산성화 진행)" if trends["pH_mean"]["slope"] < 0 else "증가"
sst_trend_txt = "상승" if trends["SST_mean"]["slope"] > 0 else "하강"
sp_trend_txt = "감소" if trends["Species_mean"]["slope"] < 0 else "증가"

st.markdown(f"""
<br>
<div class='info-box'>
📌 <b>추세 해석</b><br>
• pH: 연간 <b>{trends['pH_mean']['slope']:+.4f}</b> 변화 → 2030년까지 {ph_trend_txt} 예상<br>
• 수온: 연간 <b>{trends['SST_mean']['slope']:+.4f}°C</b> 변화 → 2030년까지 {sst_trend_txt} 예상<br>
• 관측 종 수: 연간 <b>{trends['Species_mean']['slope']:+.4f}종</b> 변화 → 2030년까지 {sp_trend_txt} 예상
</div>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PART 2. 직접 조작 시뮬레이션
# ════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🎚️ PART 2. 내가 직접 환경을 바꿔보면?</div>",
            unsafe_allow_html=True)
st.markdown("""
<div class='warning-box'>
🎮 왼쪽 사이드바의 슬라이더로 해양 환경 조건을 직접 바꿔보세요.
조건을 바꾸면 <b>예상 생물종 수가 실시간으로 변화</b>합니다.<br>
실제 데이터 범위: pH (7.87 ~ 8.20) / 수온 (23.6°C ~ 33.2°C)
</div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# 예측 실행
input_vals = np.array([[sim_ph, sim_sst, sim_sev, sim_hw_num]])
predicted_species = float(model.predict(input_vals)[0])
predicted_species = max(0.0, predicted_species)

diff_from_avg = predicted_species - current_avg
diff_pct = (diff_from_avg / current_avg) * 100

# 위험도 판별
if predicted_species >= current_avg * 0.9:
    risk_level = "안전"
    risk_color = "#2ecc71"
    risk_class = "result-safe"
    risk_emoji = "✅"
    risk_msg = "현재 수준과 비슷하게 생태계가 유지되고 있어요."
elif predicted_species >= current_avg * 0.7:
    risk_level = "주의"
    risk_color = "#f39c12"
    risk_class = "result-warning"
    risk_emoji = "⚠️"
    risk_msg = "생태계가 일부 위협받고 있어요. 환경 개선이 필요합니다."
else:
    risk_level = "위험"
    risk_color = "#e74c3c"
    risk_class = "result-danger"
    risk_emoji = "🚨"
    risk_msg = "생태계가 심각하게 위협받고 있어요! 즉각적인 환경 대응이 필요합니다."

# 결과 카드
col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    st.markdown(f"""
    <div class='result-card {risk_class}'>
        <div style='color:{risk_color}; font-size:1rem;'>예상 생물종 수</div>
        <div class='result-value' style='color:{risk_color};'>{predicted_species:.0f}종</div>
        <div class='result-label'>현재 평균: {current_avg:.0f}종</div>
    </div>""", unsafe_allow_html=True)

with col_res2:
    arrow2 = "▲" if diff_from_avg > 0 else "▼"
    color2 = "#2ecc71" if diff_from_avg > 0 else "#e74c3c"
    st.markdown(f"""
    <div class='result-card' style='background:linear-gradient(135deg,#0d2a3e,#0a3352);
         border:2px solid {color2};'>
        <div style='color:{color2}; font-size:1rem;'>현재 평균 대비</div>
        <div class='result-value' style='color:{color2};'>{arrow2}{abs(diff_pct):.1f}%</div>
        <div class='result-label'>{arrow2} {abs(diff_from_avg):.1f}종</div>
    </div>""", unsafe_allow_html=True)

with col_res3:
    st.markdown(f"""
    <div class='result-card {risk_class}'>
        <div style='color:{risk_color}; font-size:1rem;'>생태계 위험도</div>
        <div class='result-value' style='color:{risk_color};'>{risk_emoji} {risk_level}</div>
        <div class='result-label'>{risk_msg}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 게이지 차트
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=predicted_species,
    delta={"reference": current_avg, "valueformat": ".0f",
           "increasing": {"color": "#2ecc71"},
           "decreasing": {"color": "#e74c3c"}},
    gauge={
        "axis": {"range": [0, 200], "tickcolor": FONT_COLOR},
        "bar": {"color": risk_color},
        "bgcolor": PLOT_BG,
        "steps": [
            {"range": [0, current_avg * 0.7], "color": "rgba(231,76,60,0.2)"},
            {"range": [current_avg * 0.7, current_avg * 0.9], "color": "rgba(243,156,18,0.2)"},
            {"range": [current_avg * 0.9, 200], "color": "rgba(46,204,113,0.2)"},
        ],
        "threshold": {
            "line": {"color": "#ffffff", "width": 3},
            "thickness": 0.75,
            "value": current_avg
        }
    },
    title={"text": "예상 생물종 수 (흰선: 현재 평균)",
           "font": {"color": ACCENT, "size": 14}},
    number={"suffix": "종", "font": {"color": risk_color, "size": 40}}
))
fig.update_layout(
    paper_bgcolor=PAPER_BG, font=dict(color=FONT_COLOR), height=380,
    margin=dict(l=30, r=30, t=60, b=30)
)
st.plotly_chart(fig, use_container_width=True)

# pH & 수온 변화 곡선
st.markdown("<div class='section-header'>📊 pH / 수온 변화에 따른 종 수 변화 곡선</div>",
            unsafe_allow_html=True)

col_g1, col_g2 = st.columns(2)

with col_g1:
    ph_range = np.linspace(7.6, 8.4, 100)
    sp_by_ph = model.predict(
        np.column_stack([ph_range,
                         np.full(100, sim_sst),
                         np.full(100, sim_sev),
                         np.full(100, sim_hw_num)])
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ph_range, y=sp_by_ph, mode="lines",
        line=dict(color=ACCENT, width=3), name="예상 종 수"
    ))
    fig.add_vline(x=sim_ph, line_dash="dash", line_color="#ffffff", line_width=2,
                  annotation_text=f"설정 pH: {sim_ph:.2f}",
                  annotation_font_color="#ffffff")
    fig.add_hline(y=current_avg, line_dash="dot", line_color="#f39c12",
                  annotation_text=f"평균: {current_avg:.0f}종",
                  annotation_font_color="#f39c12")
    fig = style_fig(fig, "pH 변화 → 종 수 변화", height=380)
    fig.update_layout(xaxis_title="pH", yaxis_title="예상 종 수")
    st.plotly_chart(fig, use_container_width=True)

with col_g2:
    sst_range = np.linspace(22.0, 35.0, 100)
    sp_by_sst = model.predict(
        np.column_stack([np.full(100, sim_ph),
                         sst_range,
                         np.full(100, sim_sev),
                         np.full(100, sim_hw_num)])
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sst_range, y=sp_by_sst, mode="lines",
        line=dict(color="#ff6b6b", width=3), name="예상 종 수"
    ))
    fig.add_vline(x=sim_sst, line_dash="dash", line_color="#ffffff", line_width=2,
                  annotation_text=f"설정 수온: {sim_sst:.1f}°C",
                  annotation_font_color="#ffffff")
    fig.add_hline(y=current_avg, line_dash="dot", line_color="#f39c12",
                  annotation_text=f"평균: {current_avg:.0f}종",
                  annotation_font_color="#f39c12")
    fig = style_fig(fig, "수온 변화 → 종 수 변화", height=380)
    fig.update_layout(xaxis_title="수온(°C)", yaxis_title="예상 종 수")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
# PART 3. 현재 추세 vs 내 설정 비교
# ════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📊 PART 3. 현재 추세 vs 내 시뮬레이션 비교</div>",
            unsafe_allow_html=True)

comp_labels = ["현재 평균 (2015~2023)", "추세 예측 (2030년)", "내 시뮬레이션 설정값"]
comp_values = [current_avg, pred_sp[-1], predicted_species]
comp_colors = ["#00d4ff", "#f39c12", risk_color]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=comp_labels, y=comp_values,
    marker_color=comp_colors,
    text=[f"{v:.1f}종" for v in comp_values],
    textposition="outside", width=0.5
))
fig.add_hline(y=current_avg, line_dash="dot", line_color="#ffffff",
              annotation_text="현재 평균선", annotation_font_color="#ffffff")
fig = style_fig(fig, "현재 평균 vs 2030 추세 예측 vs 내 시뮬레이션 비교", height=450)
fig.update_layout(yaxis_title="예상 생물종 수", showlegend=False,
                  yaxis=dict(range=[0, max(comp_values) * 1.2]))
st.plotly_chart(fig, use_container_width=True)

# 최종 결론 (f-string 밖에서 미리 계산)
st.markdown("<div class='section-header'>💡 시뮬레이션 결론</div>",
            unsafe_allow_html=True)

ph_comment = "(현재 평균보다 낮음 → 산성화 심화)" if sim_ph < ph_avg \
    else "(현재 평균보다 높음 → 산성화 완화)"
sst_comment = "(현재 평균보다 높음 → 열파 위험 증가)" if sim_sst > sst_avg \
    else "(현재 평균보다 낮음 → 열파 위험 감소)"
diff_comment = "감소" if diff_from_avg < 0 else "증가"
model_comment = "데이터를 비교적 잘 설명하는 모델이에요." if model_score > 0.3 \
    else "데이터 변동성이 커서 예측의 불확실성이 있어요. 탐구 참고용으로 활용하세요."

st.markdown(f"""
<div class='info-box'>
🎯 <b>내 시뮬레이션 설정 요약</b><br>
• pH: <b>{sim_ph:.2f}</b> {ph_comment}<br>
• 수온: <b>{sim_sst:.1f}°C</b> {sst_comment}<br>
• 백화 심각도: <b>{sim_sev_label}</b><br>
• 해양열파: <b>{sim_hw}</b><br><br>
📊 <b>예측 결과</b>: 이 조건에서 예상 생물종 수는 <b>{predicted_species:.0f}종</b>으로,
현재 평균({current_avg:.0f}종) 대비
<b>{diff_comment} ({abs(diff_pct):.1f}%)</b>가 예상됩니다.<br><br>
🌊 <b>모델 정확도(R²)</b>: {model_score:.3f} → {model_comment}
</div>""", unsafe_allow_html=True)

# ── 하단 ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a8fa8; font-size:0.8rem; padding:10px 0;'>
🔮 미래 해양 생태계 시뮬레이션 | 당곡고등학교 해양환경 탐구 프로젝트
</div>""", unsafe_allow_html=True)
