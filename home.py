import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 페이지 기본 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="해양 산성화와 생태계 변화",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 공통 색상 테마 ────────────────────────────────────────────────
PLOT_BG    = "#0a1a2e"
PAPER_BG   = "#0d2137"
FONT_COLOR = "#c8eef5"
ACCENT     = "#00d4ff"

# Plotly 공통 레이아웃 함수 (폰트 지정 제거 → 한글 깨짐 방지)
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
    fig.update_xaxes(gridcolor="rgba(0,212,255,0.1)", zerolinecolor="rgba(0,212,255,0.2)",
                     color=FONT_COLOR)
    fig.update_yaxes(gridcolor="rgba(0,212,255,0.1)", zerolinecolor="rgba(0,212,255,0.2)",
                     color=FONT_COLOR)
    return fig


# ── CSS 스타일 ────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
    }
    .main-title {
        text-align: center; font-size: 2.6rem; font-weight: 900;
        color: #00d4ff; text-shadow: 0 0 30px rgba(0,212,255,0.5);
        padding: 20px 0 5px 0; letter-spacing: 2px;
    }
    .sub-title {
        text-align: center; font-size: 1.1rem; color: #7ecfdf; padding-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #0d2a3e, #0a3352);
        border: 1px solid #00d4ff44; border-radius: 14px;
        padding: 20px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,212,255,0.15);
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #7ecfdf; margin-top: 5px; }
    .section-header {
        color: #00d4ff; font-size: 1.4rem; font-weight: 700;
        border-left: 4px solid #00d4ff; padding-left: 12px; margin: 25px 0 15px 0;
    }
    .info-box {
        background: rgba(0,212,255,0.08); border: 1px solid #00d4ff55;
        border-radius: 10px; padding: 15px 20px; color: #c8eef5;
        font-size: 0.95rem; line-height: 1.7;
    }
    .warning-box {
        background: rgba(255,165,0,0.1); border: 1px solid #ffa50066;
        border-radius: 10px; padding: 15px 20px; color: #ffd580;
        font-size: 0.95rem; line-height: 1.7;
    }
    .danger-box {
        background: rgba(255,60,60,0.1); border: 1px solid #ff3c3c66;
        border-radius: 10px; padding: 15px 20px; color: #ffaaaa;
        font-size: 0.95rem; line-height: 1.7;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0,212,255,0.05); border-radius: 10px; padding: 5px;
    }
    .stTabs [data-baseweb="tab"] { color: #7ecfdf; font-weight: 600; }
    .stTabs [aria-selected="true"] {
        background: rgba(0,212,255,0.2) !important;
        color: #00d4ff !important; border-radius: 8px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1e30, #0d2a3e);
        border-right: 1px solid #00d4ff33;
    }
    .stMarkdown, p, span, div { color: #c8eef5; }
    h1, h2, h3 { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 로드 ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    # ⚠️ 파일명이 실제 GitHub 파일과 정확히 일치해야 합니다!
    df = pd.read_csv("realistic_ocean_climate_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    severity_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    df["Bleaching_Num"] = df["Bleaching Severity"].map(severity_map)
    df["Heatwave_Num"] = df["Marine Heatwave"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})
    sev_kr = {"None": "없음", "Low": "낮음", "Medium": "중간", "High": "높음"}
    df["백화_한글"] = df["Bleaching Severity"].map(sev_kr)
    return df

df = load_data()

# ── 사이드바 ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 필터 설정")
    st.markdown("---")
    years = sorted(df["Year"].unique())
    selected_years = st.slider(
        "📅 연도 범위",
        min_value=int(min(years)), max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )
    locations = ["전체"] + sorted(df["Location"].unique().tolist())
    selected_loc = st.selectbox("📍 관측 지역", locations)

    st.markdown("---")
    st.markdown("### 📊 데이터 요약")
    st.markdown(f"- **총 관측 수**: {len(df):,}건")
    st.markdown(f"- **관측 기간**: {df['Year'].min()} ~ {df['Year'].max()}")
    st.markdown(f"- **관측 지역**: {df['Location'].nunique()}곳")
    st.markdown(f"- **해양열파 발생**: {df['Heatwave_Num'].sum():.0f}건")

# ── 데이터 필터링 ────────────────────────────────────────────────
mask = (df["Year"] >= selected_years[0]) & (df["Year"] <= selected_years[1])
if selected_loc != "전체":
    mask &= (df["Location"] == selected_loc)
df_filtered = df[mask].copy()


# ── 헤더 ────────────────────────────────────────────────────────
st.markdown("""
<div class='main-title'>🌊 해양 산성화와 생태계 변화 분석</div>
<div class='sub-title'>Ocean Acidification & Ecosystem Decline Dashboard</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
🔬 <b>해양 산성화(Ocean Acidification)</b>란 대기 중 이산화탄소(CO₂)가 바다에 흡수되어
탄산(H₂CO₃)을 형성하면서 해수의 <b>pH가 낮아지는 현상</b>입니다.
pH가 낮을수록 산호의 골격 형성이 어려워지고, 백화 현상이 심해지며, 관측 종 수가 감소합니다.
이 대시보드는 2015~2023년 7개 주요 해역의 실제 관측 데이터를 분석합니다.
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── 주요 지표 카드 ────────────────────────────────────────────────
st.markdown("<div class='section-header'>📈 주요 지표</div>", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
avg_ph   = df_filtered["pH Level"].mean()
avg_sst  = df_filtered["SST (°C)"].mean()
avg_sp   = df_filtered["Species Observed"].mean()
high_pct = (df_filtered["Bleaching Severity"] == "High").sum() / len(df_filtered) * 100
hw_pct   = df_filtered["Heatwave_Num"].mean() * 100

cards = [
    (col1, f"{avg_ph:.3f}",  "평균 pH"),
    (col2, f"{avg_sst:.1f}°C", "평균 수면 수온"),
    (col3, f"{avg_sp:.0f}종",  "평균 관측 종 수"),
    (col4, f"{high_pct:.1f}%", "심각 백화 비율"),
    (col5, f"{hw_pct:.1f}%",   "해양열파 발생률"),
]
for col, val, label in cards:
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{val}</div>
        <div class='metric-label'>{label}</div>
    </div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── 탭 구성 ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 pH & 생태계 상관관계", "📆 연도별 추세",
    "🗺️ 지역별 비교", "🪸 백화 현상 분석", "🌡️ 해양열파 영향"
])

SEV_ORDER  = ["없음", "낮음", "중간", "높음"]
SEV_COLORS = {"없음": "#2ecc71", "낮음": "#f1c40f", "중간": "#e67e22", "높음": "#e74c3c"}


# ══════════════════════════════════════════════════════
# TAB 1 : pH & 생태계 상관관계
# ══════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════
# TAB 1 : 생물종 다양성 종합 분석 보고서
# ══════════════════════════════════════════════════════
with tab1:
    # ───────── ① 연구 질문 ─────────
    st.markdown("<div class='section-header'>① 연구 질문</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='warning-box'>
    🔬 <b>핵심 질문</b>: 해양 환경 요인(pH, 수온, 백화, 해양열파) 중
    <b>무엇이 생물종 다양성에 가장 큰 영향</b>을 줄까?<br>
    아래 데이터와 분석을 통해 단계적으로 확인해봅니다.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ───────── ② 데이터 살펴보기 ─────────
    st.markdown("<div class='section-header'>② 데이터 살펴보기</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='info-box'>
    📋 현재 필터 조건에 해당하는 <b>{len(df_filtered)}건</b>의 관측 데이터입니다.
    아래 표에서 원본 데이터를 직접 확인할 수 있어요.
    </div>""", unsafe_allow_html=True)

    show_cols = ["Date", "Location", "pH Level", "SST (°C)",
                 "Species Observed", "백화_한글", "Marine Heatwave"]
    preview = df_filtered[show_cols].rename(columns={
        "Date": "날짜", "Location": "지역", "pH Level": "pH",
        "SST (°C)": "수온(°C)", "Species Observed": "관측종수",
        "백화_한글": "백화도", "Marine Heatwave": "해양열파"
    })
    st.dataframe(preview, use_container_width=True, height=250)

    # 기초 통계
    with st.expander("📊 기초 통계 요약 보기"):
        stat_df = df_filtered[["pH Level", "SST (°C)", "Species Observed"]].describe().round(2)
        stat_df.index = ["개수", "평균", "표준편차", "최솟값",
                         "25%", "중앙값", "75%", "최댓값"]
        stat_df.columns = ["pH", "수온(°C)", "관측종수"]
        st.dataframe(stat_df, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ───────── ③ 요인별 분석 ─────────
    st.markdown("<div class='section-header'>③ 요인별 분석: 무엇이 생물종 수에 영향을 줄까?</div>",
                unsafe_allow_html=True)

    df_t1 = df_filtered.dropna(
        subset=["pH Level", "SST (°C)", "Species Observed", "Bleaching_Num"]
    ).copy()

    # 각 요인과 종 수의 상관계수 계산
    factors = {
        "pH": ("pH Level", "pH 수준"),
        "수온": ("SST (°C)", "수온(°C)"),
        "백화도": ("Bleaching_Num", "백화 심각도"),
        "해양열파": ("Heatwave_Num", "해양열파 발생"),
    }
    corr_results = {}
    for name, (col, _) in factors.items():
        r, p = stats.pearsonr(df_t1[col], df_t1["Species Observed"])
        corr_results[name] = {"r": r, "p": p}

    # 요인별 상관계수 막대그래프 (한눈에 비교!)
    st.markdown("##### 📊 각 요인이 생물종 수와 얼마나 관련 있을까? (상관계수 비교)")
    corr_df = pd.DataFrame({
        "요인": list(corr_results.keys()),
        "상관계수": [v["r"] for v in corr_results.values()]
    })
    corr_df["절댓값"] = corr_df["상관계수"].abs()
    corr_df = corr_df.sort_values("절댓값", ascending=True)

    fig = px.bar(
        corr_df, x="상관계수", y="요인", orientation="h",
        color="상관계수", color_continuous_scale="RdBu",
        range_color=[-0.5, 0.5], text="상관계수",
        labels={"상관계수": "생물종 수와의 상관계수 (r)", "요인": ""}
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.add_vline(x=0, line_color="#ffffff", line_width=1)
    fig = style_fig(fig, "막대가 길수록(좌우로) 종 다양성에 큰 영향", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='info-box'>
    💡 막대가 <b>오른쪽(+)</b>이면 "그 요인이 클수록 종 수 증가",
    <b>왼쪽(−)</b>이면 "그 요인이 클수록 종 수 감소"를 의미해요.
    막대의 <b>길이(절댓값)</b>가 길수록 영향이 큽니다.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 4개 요인 각각의 산점도/막대 (2x2)
    st.markdown("##### 🔍 요인별 상세 그래프")
    col_a, col_b = st.columns(2)

    # (1) pH 구간별 평균 종 수
    with col_a:
        df_t1["pH_구간"] = pd.cut(df_t1["pH Level"], bins=5, precision=2).astype(str)
        g = df_t1.groupby("pH_구간")["Species Observed"].mean().reset_index().sort_values("pH_구간")
        fig = px.bar(g, x="pH_구간", y="Species Observed",
                     color="Species Observed", color_continuous_scale="Tealgrn",
                     labels={"pH_구간": "pH 구간 (낮음→높음)", "Species Observed": "평균 종 수"})
        fig = style_fig(fig, "① pH 구간별 평균 종 수", height=380)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # (2) 수온 구간별 평균 종 수
    with col_b:
        df_t1["수온_구간"] = pd.cut(df_t1["SST (°C)"], bins=5, precision=1).astype(str)
        g = df_t1.groupby("수온_구간")["Species Observed"].mean().reset_index().sort_values("수온_구간")
        fig = px.bar(g, x="수온_구간", y="Species Observed",
                     color="Species Observed", color_continuous_scale="OrRd_r",
                     labels={"수온_구간": "수온 구간 (낮음→높음)", "Species Observed": "평균 종 수"})
        fig = style_fig(fig, "② 수온 구간별 평균 종 수", height=380)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    # (3) 백화도별 평균 종 수
    with col_c:
        fig = px.box(df_filtered, x="백화_한글", y="Species Observed",
                     color="백화_한글", category_orders={"백화_한글": SEV_ORDER},
                     color_discrete_map=SEV_COLORS,
                     labels={"백화_한글": "백화 심각도", "Species Observed": "관측 종 수"})
        fig = style_fig(fig, "③ 백화 심각도별 종 수", height=380)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # (4) 해양열파 유무별 평균 종 수
    with col_d:
        df_filtered["열파_한글"] = df_filtered["Heatwave_Num"].map({1: "열파 있음", 0: "열파 없음"})
        fig = px.box(df_filtered, x="열파_한글", y="Species Observed",
                     color="열파_한글",
                     color_discrete_map={"열파 없음": "#3498db", "열파 있음": "#e74c3c"},
                     labels={"열파_한글": "해양열파", "Species Observed": "관측 종 수"})
        fig = style_fig(fig, "④ 해양열파 유무별 종 수", height=380)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 상관관계 히트맵
    st.markdown("##### 🌡️ 전체 변수 상관관계 히트맵")
    numeric_cols = ["pH Level", "SST (°C)", "Species Observed",
                    "Bleaching_Num", "Heatwave_Num"]
    col_labels = ["pH", "수온", "관측종수", "백화도", "해양열파"]
    corr = df_filtered[numeric_cols].corr()
    corr.index = col_labels
    corr.columns = col_labels
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig = style_fig(fig, "변수 간 상관관계 (1에 가까울수록 강한 양의 관계)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ───────── ④ 종합 결론 ─────────
    st.markdown("<div class='section-header'>④ 종합 결론</div>", unsafe_allow_html=True)

    # 가장 영향이 큰 요인 자동 판별
    strongest = max(corr_results.items(), key=lambda x: abs(x[1]["r"]))
    strongest_name = strongest[0]
    strongest_r = strongest[1]["r"]

    # 각 요인 해석 문장 자동 생성
    def interpret(name, r):
        direction = "많아지는" if r > 0 else "줄어드는"
        strength = ("강하게" if abs(r) > 0.3 else
                    "어느 정도" if abs(r) > 0.1 else "거의 무관하게")
        return f"<b>{name}</b>: r={r:.3f} → {name}이(가) 클수록 종 수가 {strength} {direction} 경향"

    interpretations = "<br>".join(
        f"• {interpret(name, v['r'])}" for name, v in corr_results.items()
    )

    st.markdown(f"""
    <div class='danger-box'>
    📌 <b>분석 요약</b><br>
    {interpretations}<br><br>
    🎯 <b>가장 영향이 큰 요인</b>: <b>{strongest_name}</b> (상관계수 {strongest_r:.3f})
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-box'>
    🌊 <b>결론</b><br>
    1. 생물종 다양성은 <b>pH 하나만이 아니라</b> 수온·백화·해양열파 등
       <b>여러 요인이 복합적으로</b> 작용합니다.<br>
    2. 이번 데이터에서는 <b>{strongest_name}</b>이(가) 종 다양성과 가장 밀접한 관련을 보였습니다.<br>
    3. 해양 산성화(pH 하락)와 수온 상승은 <b>서로 연결</b>되어 있어,
       기후변화 대응이 곧 해양 생태계 보호로 이어집니다.<br><br>
    💭 <b>더 생각해보기</b>: 한 가지 요인만으로 결론 내리지 않고,
    여러 요인을 함께 보는 것이 왜 중요할까요?
    </div>""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════
# TAB 2 : 연도별 추세
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📆 연도별 변화 추세</div>",
                unsafe_allow_html=True)
    yearly = df_filtered.groupby("Year").agg(
        pH_mean=("pH Level", "mean"),
        SST_mean=("SST (°C)", "mean"),
        Species_mean=("Species Observed", "mean"),
        Bleaching_mean=("Bleaching_Num", "mean"),
        Heatwave_rate=("Heatwave_Num", "mean"),
    ).reset_index()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("평균 pH", "평균 수온(°C)", "평균 관측 종 수", "평균 백화 심각도")
    )
    line_cfg = [
        (1, 1, "pH_mean", "#00d4ff"),
        (1, 2, "SST_mean", "#ff6b6b"),
        (2, 1, "Species_mean", "#2ecc71"),
        (2, 2, "Bleaching_mean", "#f39c12"),
    ]
    for r, c, col, color in line_cfg:
        rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        fig.add_trace(
            go.Scatter(x=yearly["Year"], y=yearly[col], mode="lines+markers",
                       line=dict(color=color, width=3), marker=dict(size=8),
                       fill="tozeroy", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.1)",
                       name=col, showlegend=False),
            row=r, col=c
        )
    fig = style_fig(fig, "연도별 해양 환경 변화 추세 (2015–2023)", height=600)
    fig.update_annotations(font=dict(color=ACCENT, size=13))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>🌡️ 연도별 해양열파 발생률</div>",
                unsafe_allow_html=True)
    yearly["열파율"] = yearly["Heatwave_rate"] * 100
    fig = px.bar(
        yearly, x="Year", y="열파율",
        color="열파율", color_continuous_scale="YlOrRd",
        labels={"Year": "연도", "열파율": "해양열파 발생률 (%)"}
    )
    fig.add_hline(y=yearly["열파율"].mean(), line_dash="dash", line_color=ACCENT,
                  annotation_text=f"평균 {yearly['열파율'].mean():.1f}%",
                  annotation_font_color=ACCENT)
    fig = style_fig(fig, "연도별 해양열파 발생률", height=400)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 연도별 수치 데이터 보기"):
        disp = yearly[["Year", "pH_mean", "SST_mean", "Species_mean",
                       "Bleaching_mean", "열파율"]].round(3)
        disp.columns = ["연도", "평균pH", "평균수온", "평균종수", "평균백화지수", "열파율(%)"]
        st.dataframe(disp, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 3 : 지역별 비교
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🗺️ 지역별 해양 환경 비교</div>",
                unsafe_allow_html=True)
    loc_stats = df_filtered.groupby("Location").agg(
        pH_mean=("pH Level", "mean"),
        pH_min=("pH Level", "min"),
        SST_mean=("SST (°C)", "mean"),
        Species_mean=("Species Observed", "mean"),
        Bleaching_mean=("Bleaching_Num", "mean"),
        Heatwave_rate=("Heatwave_Num", "mean"),
    ).reset_index().sort_values("pH_mean")

    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=loc_stats["Location"], x=loc_stats["pH_mean"], orientation="h",
            name="평균 pH", marker_color=ACCENT, opacity=0.85
        ))
        fig.add_trace(go.Bar(
            y=loc_stats["Location"], x=loc_stats["pH_min"], orientation="h",
            name="최소 pH", marker_color="#e74c3c", opacity=0.6
        ))
        fig.add_vline(x=8.1, line_dash="dot", line_color="#f39c12",
                      annotation_text="정상 기준 8.1", annotation_font_color="#f39c12")
        fig = style_fig(fig, "지역별 pH (평균/최솟값)", height=420)
        fig.update_layout(barmode="overlay", xaxis_title="pH")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        loc_sp = loc_stats.sort_values("Species_mean")
        fig = px.bar(
            loc_sp, y="Location", x="Species_mean", orientation="h",
            color="Species_mean", color_continuous_scale="YlGn",
            labels={"Location": "", "Species_mean": "평균 관측 종 수"}
        )
        fig = style_fig(fig, "지역별 평균 관측 종 수", height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>🕸️ 지역별 종합 환경 지표 (레이더 차트)</div>",
                unsafe_allow_html=True)
    categories = ["평균 pH", "평균 수온", "관측 종 수", "백화 심각도", "열파 발생률"]

    def normalize(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx != mn else s * 0 + 0.5

    loc_stats["pH_n"]  = normalize(loc_stats["pH_mean"])
    loc_stats["SST_n"] = normalize(loc_stats["SST_mean"])
    loc_stats["Sp_n"]  = normalize(loc_stats["Species_mean"])
    loc_stats["Bl_n"]  = normalize(loc_stats["Bleaching_mean"])
    loc_stats["HW_n"]  = normalize(loc_stats["Heatwave_rate"])

    fig = go.Figure()
    palette = ["#00d4ff", "#ff6b6b", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c", "#1abc9c"]
    loc_list = loc_stats.reset_index(drop=True)
    for idx, row in loc_list.iterrows():
        vals = [row["pH_n"], row["SST_n"], row["Sp_n"], row["Bl_n"], row["HW_n"]]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", name=row["Location"],
            line_color=palette[idx % len(palette)], opacity=0.7
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(visible=True, range=[0, 1], color=FONT_COLOR,
                            gridcolor="rgba(0,212,255,0.15)"),
            angularaxis=dict(color=FONT_COLOR, gridcolor="rgba(0,212,255,0.15)")
        ),
        paper_bgcolor=PAPER_BG, font=dict(color=FONT_COLOR),
        height=550, title=dict(text="지역별 환경 지표 레이더 차트 (정규화)",
                               font=dict(color=ACCENT, size=15)),
        legend=dict(bgcolor="rgba(13,33,55,0.6)")
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 4 : 백화 현상 분석
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>🪸 산호 백화 현상 심층 분석</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='danger-box'>
    🚨 <b>산호 백화(Coral Bleaching)</b>: 수온 상승과 pH 하락이 복합적으로 작용하면
    산호가 체내 공생 조류(Zooxanthellae)를 방출하여 하얗게 변하는 현상입니다.
    백화 후 회복되지 못하면 산호는 죽게 되며, 이는 전체 해양 생태계 붕괴로 이어집니다.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        sev_counts = df_filtered["백화_한글"].value_counts().reindex(SEV_ORDER).dropna()
        fig = px.pie(
            values=sev_counts.values, names=sev_counts.index, hole=0.45,
            color=sev_counts.index, color_discrete_map=SEV_COLORS
        )
        fig.update_traces(textinfo="percent+label", textfont_size=12)
        fig = style_fig(fig, "백화 심각도 분포", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.violin(
            df_filtered, x="백화_한글", y="SST (°C)", color="백화_한글",
            box=True, category_orders={"백화_한글": SEV_ORDER},
            color_discrete_map=SEV_COLORS,
            labels={"백화_한글": "백화 심각도", "SST (°C)": "수온(°C)"}
        )
        fig = style_fig(fig, "백화 심각도별 수온 분포", height=420)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>📍 지역별 백화 심각도 분포</div>",
                unsafe_allow_html=True)
    bleach_loc = pd.crosstab(df_filtered["Location"], df_filtered["백화_한글"],
                             normalize="index") * 100
    for c in SEV_ORDER:
        if c not in bleach_loc.columns:
            bleach_loc[c] = 0
    bleach_loc = bleach_loc[SEV_ORDER].reset_index()

    fig = go.Figure()
    for sev in SEV_ORDER:
        fig.add_trace(go.Bar(
            x=bleach_loc["Location"], y=bleach_loc[sev], name=sev,
            marker_color=SEV_COLORS[sev]
        ))
    fig = style_fig(fig, "지역별 백화 심각도 분포 (%)", height=450)
    fig.update_layout(barmode="stack", xaxis_title="지역", yaxis_title="비율(%)")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 5 : 해양열파 영향
# ══════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>🌡️ 해양열파(Marine Heatwave) 영향 분석</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='warning-box'>
    🔥 <b>해양열파</b>는 해수면 온도가 비정상적으로 높은 상태가 지속되는 현상입니다.
    pH 하락과 복합 작용하여 산호 백화를 급격히 가속시킵니다.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    df_filtered["열파_한글"] = df_filtered["Heatwave_Num"].map({1: "열파 있음", 0: "열파 없음"})
    hw_yes = df_filtered[df_filtered["Heatwave_Num"] == 1]
    hw_no  = df_filtered[df_filtered["Heatwave_Num"] == 0]

    col_a, col_b, col_c = st.columns(3)
    def diff_card(col, label, vy, vn, unit="", invert=False):
        diff = vy - vn
        arrow = "▲" if diff > 0 else "▼"
        color = "#e74c3c" if (diff > 0) != invert else "#2ecc71"
        col.markdown(f"""
        <div class='metric-card'>
            <div style='color:#7ecfdf;font-size:0.8rem;'>{label}</div>
            <div style='color:#e74c3c;font-size:1.4rem;font-weight:bold;'>{vy:.2f}{unit}</div>
            <div style='color:#2ecc71;font-size:0.9rem;'>(열파 없음: {vn:.2f}{unit})</div>
            <div style='color:{color};font-size:1rem;font-weight:bold;'>{arrow} {abs(diff):.2f}{unit}</div>
        </div>""", unsafe_allow_html=True)
    diff_card(col_a, "🌡️ 평균 수온", hw_yes["SST (°C)"].mean(), hw_no["SST (°C)"].mean(), "°C")
    diff_card(col_b, "🧪 평균 pH", hw_yes["pH Level"].mean(), hw_no["pH Level"].mean(), "", invert=True)
    diff_card(col_c, "🐠 평균 종 수", hw_yes["Species Observed"].mean(), hw_no["Species Observed"].mean(), "종", invert=True)
    st.markdown("<br>", unsafe_allow_html=True)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("수온(°C)", "pH", "관측 종 수"))
    compare = [("SST (°C)", 1), ("pH Level", 2), ("Species Observed", 3)]
    for col_name, c in compare:
        fig.add_trace(go.Histogram(
            x=hw_no[col_name], name="열파 없음", marker_color="#3498db",
            opacity=0.6, histnorm="probability density", showlegend=(c == 1)
        ), row=1, col=c)
        fig.add_trace(go.Histogram(
            x=hw_yes[col_name], name="열파 있음", marker_color="#e74c3c",
            opacity=0.6, histnorm="probability density", showlegend=(c == 1)
        ), row=1, col=c)
    fig = style_fig(fig, "해양열파 유무별 환경 지표 비교", height=420)
    fig.update_layout(barmode="overlay")
    fig.update_annotations(font=dict(color=ACCENT, size=13))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>📅 월별·연도별 해양열파 발생 히트맵</div>",
                unsafe_allow_html=True)
    heatmap_data = df_filtered.groupby(["Year", "Month"])["Heatwave_Num"].mean().unstack(fill_value=0)
    month_labels = ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"]
    fig = px.imshow(
        heatmap_data.values, text_auto=".2f",
        x=[month_labels[c-1] for c in heatmap_data.columns],
        y=[str(y) for y in heatmap_data.index],
        color_continuous_scale="YlOrRd", aspect="auto",
        labels={"x": "월", "y": "연도", "color": "발생률"}
    )
    fig = style_fig(fig, "월별·연도별 해양열파 발생률", height=400)
    st.plotly_chart(fig, use_container_width=True)


# ── 하단 정보 ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a8fa8; font-size:0.85rem; padding:15px 0;'>
🌊 Ocean Acidification & Ecosystem Decline Dashboard |
데이터 출처: 해양 기후 관측 데이터셋 (2015–2023) |
made by 20112박하민, 20515윤혁빈, 20327홍의찬
</div>
""", unsafe_allow_html=True)
