
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
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

# ── 한글 폰트 설정 ────────────────────────────────────────────────
@st.cache_resource
def set_korean_font():
    """시스템에서 한글 폰트를 찾아 설정"""
    font_candidates = [
        "NanumGothic", "NanumBarunGothic", "Malgun Gothic",
        "AppleGothic", "Noto Sans CJK KR", "DejaVu Sans"
    ]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in font_candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            return font
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"

font_name = set_korean_font()

# ── CSS 스타일 ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
    }
    /* 메인 타이틀 */
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 900;
        color: #00d4ff;
        text-shadow: 0 0 30px rgba(0,212,255,0.5);
        padding: 20px 0 5px 0;
        letter-spacing: 2px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #7ecfdf;
        padding-bottom: 20px;
    }
    /* 카드 스타일 */
    .metric-card {
        background: linear-gradient(135deg, #0d2a3e, #0a3352);
        border: 1px solid #00d4ff44;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,212,255,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #7ecfdf;
        margin-top: 5px;
    }
    /* 섹션 제목 */
    .section-header {
        color: #00d4ff;
        font-size: 1.4rem;
        font-weight: 700;
        border-left: 4px solid #00d4ff;
        padding-left: 12px;
        margin: 25px 0 15px 0;
    }
    /* 정보 박스 */
    .info-box {
        background: rgba(0,212,255,0.08);
        border: 1px solid #00d4ff55;
        border-radius: 10px;
        padding: 15px 20px;
        color: #c8eef5;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .warning-box {
        background: rgba(255,165,0,0.1);
        border: 1px solid #ffa50066;
        border-radius: 10px;
        padding: 15px 20px;
        color: #ffd580;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .danger-box {
        background: rgba(255,60,60,0.1);
        border: 1px solid #ff3c3c66;
        border-radius: 10px;
        padding: 15px 20px;
        color: #ffaaaa;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0,212,255,0.05);
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #7ecfdf;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,212,255,0.2) !important;
        color: #00d4ff !important;
        border-radius: 8px;
    }
    /* 사이드바 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1e30, #0d2a3e);
        border-right: 1px solid #00d4ff33;
    }
    /* 일반 텍스트 */
    .stMarkdown, p, span, div {
        color: #c8eef5;
    }
    h1, h2, h3 { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 로드 ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("realistic_ocean_climate_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    # 백화 심각도 수치 변환
    severity_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    df["Bleaching_Num"] = df["Bleaching Severity"].map(severity_map)
    # 해양열파 수치 변환
    df["Heatwave_Num"] = df["Marine Heatwave"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})
    return df

df = load_data()

# ── 사이드바 ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 필터 설정")
    st.markdown("---")

    years = sorted(df["Year"].unique())
    selected_years = st.slider(
        "📅 연도 범위",
        min_value=int(min(years)),
        max_value=int(max(years)),
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

    st.markdown("---")
    st.markdown("### 🎨 차트 색상")
    chart_color = st.selectbox(
        "테마 선택",
        ["ocean", "plasma", "viridis", "coolwarm"],
        index=0
    )

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

avg_ph     = df_filtered["pH Level"].mean()
avg_sst    = df_filtered["SST (°C)"].mean()
avg_sp     = df_filtered["Species Observed"].mean()
high_pct   = (df_filtered["Bleaching Severity"] == "High").sum() / len(df_filtered) * 100
hw_pct     = df_filtered["Heatwave_Num"].mean() * 100

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{avg_ph:.3f}</div>
        <div class='metric-label'>평균 pH</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{avg_sst:.1f}°C</div>
        <div class='metric-label'>평균 수면 수온</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{avg_sp:.0f}종</div>
        <div class='metric-label'>평균 관측 종 수</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{high_pct:.1f}%</div>
        <div class='metric-label'>심각 백화 비율</div>
    </div>""", unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{hw_pct:.1f}%</div>
        <div class='metric-label'>해양열파 발생률</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 탭 구성 ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 pH & 생태계 상관관계",
    "📆 연도별 추세",
    "🗺️ 지역별 비교",
    "🪸 백화 현상 분석",
    "🌡️ 해양열파 영향"
])


# ══════════════════════════════════════════════════════
# TAB 1 : pH & 생태계 상관관계
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>🔬 pH와 생태계 지표 간 상관관계</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='warning-box'>
    ⚠️ <b>핵심 원리</b>: pH가 <b>낮을수록</b>(산성도 증가) 해양 생태계가 파괴됩니다.<br>
    산호와 조개류는 탄산칼슘(CaCO₃)으로 골격을 만드는데, pH가 낮으면 이 과정이 방해받습니다.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # ① pH vs 관측 종 수 산점도
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#0d2137")
        ax.set_facecolor("#0a1a2e")

        scatter = ax.scatter(
            df_filtered["pH Level"],
            df_filtered["Species Observed"],
            c=df_filtered["SST (°C)"],
            cmap="YlOrRd",
            alpha=0.7,
            s=40,
            edgecolors="none"
        )
        # 추세선
        x = df_filtered["pH Level"].values
        y = df_filtered["Species Observed"].values
        mask_valid = ~np.isnan(x) & ~np.isnan(y)
        if mask_valid.sum() > 2:
            slope, intercept, r, p, _ = stats.linregress(x[mask_valid], y[mask_valid])
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="#00d4ff", linewidth=2, linestyle="--",
                    label=f"추세선 (r={r:.3f})")
            ax.legend(facecolor="#0d2137", labelcolor="white", fontsize=9)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("수온 (°C)", color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_xlabel("pH 수준", color="#7ecfdf", fontsize=11)
        ax.set_ylabel("관측 종 수", color="#7ecfdf", fontsize=11)
        ax.set_title("pH ↔ 관측 종 수 관계", color="#00d4ff", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#7ecfdf")
        for spine in ax.spines.values():
            spine.set_edgecolor("#00d4ff44")
        st.pyplot(fig)
        plt.close()

    # ② pH vs 백화 심각도 박스플롯
    with col_b:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#0d2137")
        ax.set_facecolor("#0a1a2e")

        order  = ["None", "Low", "Medium", "High"]
        colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

        for i, (sev, color) in enumerate(zip(order, colors)):
            data = df_filtered[df_filtered["Bleaching Severity"] == sev]["pH Level"].dropna()
            if len(data) > 0:
                bp = ax.boxplot(data, positions=[i], widths=0.5,
                                patch_artist=True,
                                boxprops=dict(facecolor=color, alpha=0.7),
                                medianprops=dict(color="white", linewidth=2),
                                whiskerprops=dict(color="#7ecfdf"),
                                capprops=dict(color="#7ecfdf"),
                                flierprops=dict(marker="o", color=color,
                                                markerfacecolor=color, alpha=0.5, markersize=4))

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["없음", "낮음", "중간", "높음"], color="#7ecfdf", fontsize=10)
        ax.set_xlabel("백화 심각도", color="#7ecfdf", fontsize=11)
        ax.set_ylabel("pH 수준", color="#7ecfdf", fontsize=11)
        ax.set_title("백화 심각도별 pH 분포", color="#00d4ff", fontsize=13, fontweight="bold")
        ax.tick_params(axis="y", colors="#7ecfdf")
        for spine in ax.spines.values():
            spine.set_edgecolor("#00d4ff44")
        st.pyplot(fig)
        plt.close()

    # ③ 상관관계 히트맵
    st.markdown("<div class='section-header'>📊 변수 간 상관관계 히트맵</div>",
                unsafe_allow_html=True)

    numeric_cols = ["pH Level", "SST (°C)", "Species Observed",
                    "Bleaching_Num", "Heatwave_Num"]
    col_labels   = ["pH", "수온(SST)", "관측 종 수", "백화 심각도", "해양열파"]
    corr = df_filtered[numeric_cols].corr()
    corr.index   = col_labels
    corr.columns = col_labels

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d2137")
    ax.set_facecolor("#0a1a2e")

    sns.heatmap(
        corr, annot=True, fmt=".3f",
        cmap="coolwarm", center=0,
        ax=ax, linewidths=0.5,
        annot_kws={"size": 11, "color": "white"},
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("상관관계 히트맵 (Pearson r)", color="#00d4ff",
                 fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="#7ecfdf", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#00d4ff44")
    st.pyplot(fig)
    plt.close()

    # 해석
    r_ph_sp, p_ph_sp = stats.pearsonr(
        df_filtered["pH Level"].dropna(),
        df_filtered["Species Observed"][df_filtered["pH Level"].notna()]
    )
    r_ph_bl, p_ph_bl = stats.pearsonr(
        df_filtered["pH Level"].dropna(),
        df_filtered["Bleaching_Num"][df_filtered["pH Level"].notna()]
    )
    st.markdown(f"""
    <div class='info-box'>
    📌 <b>통계 해석</b><br>
    • pH ↔ 관측 종 수: r = <b>{r_ph_sp:.3f}</b>
      (p = {p_ph_sp:.4f}) → {"유의미한 양의 상관관계" if r_ph_sp > 0.1 and p_ph_sp < 0.05 else "낮은 상관관계"}<br>
    • pH ↔ 백화 심각도: r = <b>{r_ph_bl:.3f}</b>
      (p = {p_ph_bl:.4f}) → {"유의미한 음의 상관관계 (pH 낮을수록 백화 심각)" if r_ph_bl < -0.1 and p_ph_bl < 0.05 else "확인 필요"}<br>
    • 해양 산성화(pH 감소)는 산호 백화를 유발하고 생물 다양성을 감소시킵니다.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 2 : 연도별 추세
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📆 연도별 변화 추세</div>",
                unsafe_allow_html=True)

    yearly = df_filtered.groupby("Year").agg(
        pH_mean    = ("pH Level",          "mean"),
        SST_mean   = ("SST (°C)",          "mean"),
        Species_mean = ("Species Observed","mean"),
        Bleaching_mean = ("Bleaching_Num", "mean"),
        Heatwave_rate  = ("Heatwave_Num",  "mean"),
        Count      = ("pH Level",          "count")
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#0d2137")
    fig.suptitle("연도별 해양 환경 변화 추세 (2015–2023)",
                 color="#00d4ff", fontsize=15, fontweight="bold", y=1.01)

    plot_cfg = [
        (axes[0,0], "pH_mean",        "평균 pH",        "#00d4ff", "pH 수준"),
        (axes[0,1], "SST_mean",       "평균 수온 (°C)", "#ff6b6b", "온도 (°C)"),
        (axes[1,0], "Species_mean",   "평균 관측 종 수","#2ecc71", "종 수"),
        (axes[1,1], "Bleaching_mean", "평균 백화 심각도","#f39c12","백화 지수"),
    ]

    for ax, col, title, color, ylabel in plot_cfg:
        ax.set_facecolor("#0a1a2e")
        ax.plot(yearly["Year"], yearly[col],
                color=color, linewidth=2.5, marker="o", markersize=6)
        # 추세선
        if len(yearly) > 2:
            slope, intercept, r, p, _ = stats.linregress(yearly["Year"], yearly[col])
            x_line = np.linspace(yearly["Year"].min(), yearly["Year"].max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color=color, linewidth=1.2, linestyle="--", alpha=0.6,
                    label=f"추세 (r={r:.3f})")
            ax.legend(facecolor="#0d2137", labelcolor="white", fontsize=8)
        ax.fill_between(yearly["Year"], yearly[col],
                        alpha=0.15, color=color)
        ax.set_title(title, color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel("연도", color="#7ecfdf", fontsize=9)
        ax.set_ylabel(ylabel, color="#7ecfdf", fontsize=9)
        ax.tick_params(colors="#7ecfdf", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#00d4ff33")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 해양열파 발생률 바 차트
    st.markdown("<div class='section-header'>🌡️ 연도별 해양열파 발생률</div>",
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d2137")
    ax.set_facecolor("#0a1a2e")

    bar_colors = ["#e74c3c" if v > 0.3 else "#f39c12" if v > 0.15 else "#3498db"
                  for v in yearly["Heatwave_rate"]]
    ax.bar(yearly["Year"], yearly["Heatwave_rate"] * 100,
           color=bar_colors, alpha=0.85, width=0.6, edgecolor="#ffffff22")
    ax.axhline(y=yearly["Heatwave_rate"].mean() * 100,
               color="#00d4ff", linewidth=1.5, linestyle="--",
               label=f"평균 {yearly['Heatwave_rate'].mean()*100:.1f}%")
    ax.set_xlabel("연도", color="#7ecfdf", fontsize=10)
    ax.set_ylabel("해양열파 발생률 (%)", color="#7ecfdf", fontsize=10)
    ax.set_title("연도별 해양열파 발생률", color="#00d4ff", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#7ecfdf")
    ax.legend(facecolor="#0d2137", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#00d4ff44")
    st.pyplot(fig)
    plt.close()

    # 연도별 데이터 테이블
    with st.expander("📋 연도별 수치 데이터 보기"):
        display_df = yearly.copy()
        display_df.columns = ["연도", "평균pH", "평균수온",
                               "평균종수", "평균백화지수", "열파발생률", "관측수"]
        display_df = display_df.round(4)
        st.dataframe(display_df, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 3 : 지역별 비교
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🗺️ 지역별 해양 환경 비교</div>",
                unsafe_allow_html=True)

    loc_stats = df_filtered.groupby("Location").agg(
        pH_mean        = ("pH Level",          "mean"),
        pH_min         = ("pH Level",          "min"),
        SST_mean       = ("SST (°C)",          "mean"),
        Species_mean   = ("Species Observed",  "mean"),
        Bleaching_mean = ("Bleaching_Num",     "mean"),
        Heatwave_rate  = ("Heatwave_Num",      "mean"),
        Count          = ("pH Level",          "count")
    ).reset_index().sort_values("pH_mean")

    # ① 지역별 pH 평균 + 최솟값 바 차트
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#0d2137")
        ax.set_facecolor("#0a1a2e")

        y_pos = range(len(loc_stats))
        bars  = ax.barh(y_pos, loc_stats["pH_mean"],
                        color="#00d4ff", alpha=0.8, height=0.5, label="평균 pH")
        ax.barh(y_pos, loc_stats["pH_min"],
                color="#e74c3c", alpha=0.5, height=0.5, label="최소 pH")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(loc_stats["Location"], color="#7ecfdf", fontsize=9)
        ax.set_xlabel("pH", color="#7ecfdf", fontsize=10)
        ax.set_title("지역별 pH (평균/최솟값)", color="#00d4ff",
                     fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", colors="#7ecfdf")
        ax.legend(facecolor="#0d2137", labelcolor="white", fontsize=8)
        ax.axvline(x=8.1, color="#f39c12", linewidth=1.2, linestyle=":",
                   label="정상 기준선 (8.1)")
        for spine in ax.spines.values():
            spine.set_edgecolor("#00d4ff44")
        for bar, val in zip(bars, loc_stats["pH_mean"]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", color="white", fontsize=8)
        st.pyplot(fig)
        plt.close()

    # ② 지역별 관측 종 수
    with col_b:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#0d2137")
        ax.set_facecolor("#0a1a2e")

        loc_sp = df_filtered.groupby("Location")["Species Observed"].mean().sort_values()
        colors_sp = plt.cm.YlGn(np.linspace(0.3, 0.9, len(loc_sp)))
        bars = ax.barh(range(len(loc_sp)), loc_sp.values,
                       color=colors_sp, alpha=0.85, height=0.5)
        ax.set_yticks(range(len(loc_sp)))
        ax.set_yticklabels(loc_sp.index, color="#7ecfdf", fontsize=9)
        ax.set_xlabel("평균 관측 종 수", color="#7ecfdf", fontsize=10)
        ax.set_title("지역별 평균 관측 종 수", color="#00d4ff",
                     fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", colors="#7ecfdf")
        for spine in ax.spines.values():
            spine.set_edgecolor("#00d4ff44")
        for bar, val in zip(bars, loc_sp.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", color="white", fontsize=8)
        st.pyplot(fig)
        plt.close()

    # ③ 레이더 차트 – 지역별 다차원 비교
    st.markdown("<div class='section-header'>🕸️ 지역별 종합 환경 지표 (레이더 차트)</div>",
                unsafe_allow_html=True)

    from matplotlib.patches import FancyArrowPatch

    categories = ["평균 pH", "평균 수온", "관측 종 수", "백화 심각도", "열파 발생률"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0d2137")
    ax.set_facecolor("#0a1628")

    # 정규화
    def normalize_col(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return series * 0 + 0.5
        return (series - mn) / (mx - mn)

    loc_stats["pH_norm"]      = normalize_col(loc_stats["pH_mean"])
    loc_stats["SST_norm"]     = normalize_col(loc_stats["SST_mean"])
    loc_stats["Species_norm"] = normalize_col(loc_stats["Species_mean"])
    loc_stats["Bleach_norm"]  = normalize_col(loc_stats["Bleaching_mean"])
    loc_stats["HW_norm"]      = normalize_col(loc_stats["Heatwave_rate"])

    radar_colors = ["#00d4ff","#ff6b6b","#2ecc71","#f39c12","#9b59b6","#e74c3c","#1abc9c"]

    for i, row in loc_stats.iterrows():
        vals = [row["pH_norm"], row["SST_norm"], row["Species_norm"],
                row["Bleach_norm"], row["HW_norm"]]
        vals += vals[:1]
        color = radar_colors[i % len(radar_colors)]
        ax.plot(angles, vals, "o-", linewidth=2, color=color,
                label=row["Location"])
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#7ecfdf", size=10)
    ax.set_yticklabels([], color="transparent")
    ax.grid(color="#00d4ff22")
    ax.spines["polar"].set_color("#00d4ff44")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1),
              facecolor="#0d2137", labelcolor="white", fontsize=9)
    ax.set_title("지역별 환경 지표 레이더 차트 (정규화)",
                 color="#00d4ff", fontsize=13, fontweight="bold", pad=20)
    st.pyplot(fig)
    plt.close()


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
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # ① 백화 심각도 비율 파이차트
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0d2137")
        ax.set_facecolor("#0d2137")

        sev_counts = df_filtered["Bleaching Severity"].value_counts()
        order_pie  = [s for s in ["None","Low","Medium","High"] if s in sev_counts.index]
        sizes      = [sev_counts[s] for s in order_pie]
        colors_pie = ["#2ecc71","#f1c40f","#e67e22","#e74c3c"]
        labels_pie = ["없음","낮음","중간","높음"]
        used_labels = [labels_pie[["None","Low","Medium","High"].index(s)]
                       for s in order_pie]
        used_colors = [colors_pie[["None","Low","Medium","High"].index(s)]
                       for s in order_pie]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=used_labels, autopct="%1.1f%%",
            colors=used_colors, startangle=90,
            pctdistance=0.85, textprops={"color":"white","fontsize":10},
            wedgeprops={"edgecolor":"#0d2137","linewidth":2},
            shadow=True
        )
        for autotext in autotexts:
            autotext.set_fontsize(9)
        ax.set_title("백화 심각도 분포", color="#00d4ff",
                     fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close()

    # ② 백화 심각도 × 수온/pH 바이올린
    with col_b:
        fig, axes = plt.subplots(1, 2, figsize=(7, 5))
        fig.patch.set_facecolor("#0d2137")

        order_v = ["None","Low","Medium","High"]
        pal_v   = {"None":"#2ecc71","Low":"#f1c40f","Medium":"#e67e22","High":"#e74c3c"}

        for ax_v, col_v, ylabel_v, title_v in [
            (axes[0], "SST (°C)",  "수온 (°C)", "백화도별 수온"),
            (axes[1], "pH Level",  "pH",         "백화도별 pH")
        ]:
            ax_v.set_facecolor("#0a1a2e")
            data_list  = [df_filtered[df_filtered["Bleaching Severity"]==s][col_v].dropna()
                          for s in order_v]
            data_list  = [d for d in data_list if len(d) > 0]
            valid_order = [s for s, d in zip(order_v, [
                df_filtered[df_filtered["Bleaching Severity"]==s][col_v].dropna()
                for s in order_v]) if len(d) > 0]

            parts = ax_v.violinplot(data_list, showmeans=True, showmedians=True)
            for i, (body, sev) in enumerate(zip(parts["bodies"], valid_order)):
                body.set_facecolor(pal_v[sev])
                body.set_alpha(0.7)
            parts["cmeans"].set_color("white")
            parts["cmedians"].set_color("#00d4ff")
            parts["cbars"].set_color("#00d4ff44")
            parts["cmaxes"].set_color("#00d4ff44")
            parts["cmins"].set_color("#00d4ff44")

            ax_v.set_xticks(range(1, len(valid_order)+1))
            ax_v.set_xticklabels(
                [{"None":"없음","Low":"낮음","Medium":"중간","High":"높음"}[s]
                 for s in valid_order],
                color="#7ecfdf", fontsize=8
            )
            ax_v.set_ylabel(ylabel_v, color="#7ecfdf", fontsize=9)
            ax_v.set_title(title_v, color="#00d4ff", fontsize=10, fontweight="bold")
            ax_v.tick_params(axis="y", colors="#7ecfdf")
            for spine in ax_v.spines.values():
                spine.set_edgecolor("#00d4ff44")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ③ 지역별 백화 심각도 스택 바
    st.markdown("<div class='section-header'>📍 지역별 백화 심각도 분포</div>",
                unsafe_allow_html=True)

    bleach_loc = pd.crosstab(
        df_filtered["Location"],
        df_filtered["Bleaching Severity"],
        normalize="index"
    ) * 100

    for col_missing in ["None","Low","Medium","High"]:
        if col_missing not in bleach_loc.columns:
            bleach_loc[col_missing] = 0
    bleach_loc = bleach_loc[["None","Low","Medium","High"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d2137")
    ax.set_facecolor("#0a1a2e")

    bar_colors_stack = ["#2ecc71","#f1c40f","#e67e22","#e74c3c"]
    bar_labels_stack = ["없음","낮음","중간","높음"]
    bottom = np.zeros(len(bleach_loc))

    for color_s, label_s, col_s in zip(bar_colors_stack, bar_labels_stack,
                                        ["None","Low","Medium","High"]):
        ax.bar(bleach_loc.index, bleach_loc[col_s],
               bottom=bottom, label=label_s,
               color=color_s, alpha=0.85, edgecolor="#0d2137", linewidth=0.5)
        bottom += bleach_loc[col_s].values

    ax.set_xlabel("지역", color="#7ecfdf", fontsize=10)
    ax.set_ylabel("비율 (%)", color="#7ecfdf", fontsize=10)
    ax.set_title("지역별 백화 심각도 분포 (%)", color="#00d4ff",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="#7ecfdf")
    ax.legend(facecolor="#0d2137", labelcolor="white", fontsize=9,
              loc="upper right")
    for spine in ax.spines.values():
        spine.set_edgecolor("#00d4ff44")
    plt.xticks(rotation=25, ha="right")
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════
# TAB 5 : 해양열파 영향
# ══════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>🌡️ 해양열파(Marine Heatwave) 영향 분석</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='warning-box'>
    🔥 <b>해양열파</b>는 해수면 온도가 비정상적으로 높은 상태가 5일 이상 지속되는 현상입니다.
    해양열파는 pH 하락과 복합 작용하여 산호 백화를 급격히 가속시킵니다.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    hw_yes = df_filtered[df_filtered["Heatwave_Num"] == 1]
    hw_no  = df_filtered[df_filtered["Heatwave_Num"] == 0]

    col_a, col_b, col_c = st.columns(3)

    def diff_card(col, label, val_yes, val_no, unit="", invert=False):
        diff = val_yes - val_no
        arrow = "▲" if diff > 0 else "▼"
        color = "#e74c3c" if (diff > 0) != invert else "#2ecc71"
        col.markdown(f"""
        <div class='metric-card'>
            <div style='color:#7ecfdf;font-size:0.8rem;'>{label}</div>
            <div style='color:#e74c3c;font-size:1.4rem;font-weight:bold;'>
                {val_yes:.2f}{unit}</div>
            <div style='color:#2ecc71;font-size:0.9rem;'>(열파 없음: {val_no:.2f}{unit})</div>
            <div style='color:{color};font-size:1rem;font-weight:bold;'>
                {arrow} {abs(diff):.2f}{unit}</div>
        </div>""", unsafe_allow_html=True)

    diff_card(col_a, "🌡️ 평균 수온",
              hw_yes["SST (°C)"].mean(), hw_no["SST (°C)"].mean(), "°C")
    diff_card(col_b, "🧪 평균 pH",
              hw_yes["pH Level"].mean(), hw_no["pH Level"].mean(), "", invert=True)
    diff_card(col_c, "🐠 평균 관측 종 수",
              hw_yes["Species Observed"].mean(), hw_no["Species Observed"].mean(), "종",
              invert=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 해양열파 유무별 수온·pH 분포 비교
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d2137")
    fig.suptitle("해양열파 유무별 환경 지표 비교",
                 color="#00d4ff", fontsize=14, fontweight="bold")

    compare_cols = [("SST (°C)", "수온 (°C)"), ("pH Level", "pH"),
                    ("Species Observed", "관측 종 수")]

    for ax_c, (col_c_name, ylabel_c) in zip(axes, compare_cols):
        ax_c.set_facecolor("#0a1a2e")
        data_yes = hw_yes[col_c_name].dropna()
        data_no  = hw_no[col_c_name].dropna()

        ax_c.hist(data_no,  bins=20, color="#3498db", alpha=0.6,
                  label="열파 없음", density=True, edgecolor="none")
        ax_c.hist(data_yes, bins=20, color="#e74c3c", alpha=0.6,
                  label="열파 있음", density=True, edgecolor="none")
        ax_c.axvline(data_no.mean(),  color="#3498db",
                     linewidth=2, linestyle="--", alpha=0.9)
        ax_c.axvline(data_yes.mean(), color="#e74c3c",
                     linewidth=2, linestyle="--", alpha=0.9)

        ax_c.set_xlabel(ylabel_c, color="#7ecfdf", fontsize=10)
        ax_c.set_ylabel("밀도", color="#7ecfdf", fontsize=10)
        ax_c.set_title(ylabel_c, color="#00d4ff", fontsize=11, fontweight="bold")
        ax_c.tick_params(colors="#7ecfdf")
        ax_c.legend(facecolor="#0d2137", labelcolor="white", fontsize=8)
        for spine in ax_c.spines.values():
            spine.set_edgecolor("#00d4ff44")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 월별 해양열파 발생 히트맵
    st.markdown("<div class='section-header'>📅 월별·연도별 해양열파 발생 히트맵</div>",
                unsafe_allow_html=True)

    heatmap_data = df_filtered.groupby(["Year","Month"])["Heatwave_Num"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0d2137")
    ax.set_facecolor("#0a1a2e")

    month_labels = ["1월","2월","3월","4월","5월","6월",
                    "7월","8월","9월","10월","11월","12월"]
    col_labels_hm = [month_labels[c-1] for c in heatmap_data.columns]

    sns.heatmap(
        heatmap_data, annot=True, fmt=".2f",
        cmap="YlOrRd", ax=ax,
        linewidths=0.3, linecolor="#0d2137",
        xticklabels=col_labels_hm,
        annot_kws={"size":8, "color":"white"},
        cbar_kws={"label":"발생률", "shrink":0.8}
    )
    ax.set_title("월별·연도별 해양열파 발생률", color="#00d4ff",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("월", color="#7ecfdf", fontsize=10)
    ax.set_ylabel("연도", color="#7ecfdf", fontsize=10)
    ax.tick_params(axis="x", colors="#7ecfdf", rotation=0)
    ax.tick_params(axis="y", colors="#7ecfdf", rotation=0)
    st.pyplot(fig)
    plt.close()


# ── 하단 정보 ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a8fa8; font-size:0.85rem; padding:15px 0;'>
🌊 Ocean Acidification & Ecosystem Decline Dashboard |
데이터 출처: 해양 기후 관측 데이터셋 (2015–2023) |
당곡고등학교 해양환경 프로젝트
</div>
""", unsafe_allow_html=True)
