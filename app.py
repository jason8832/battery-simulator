import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
from sklearn.ensemble import RandomForestRegressor

# --- [1] 페이지 기본 설정 (반드시 가장 윗줄) ---
st.set_page_config(page_title="Battery AI Simulator", layout="wide", page_icon="🔋")

# ==============================================================================
# [0] 디자인 & CSS 설정
# ==============================================================================

def get_img_tag(file, title, css_class="logo-img"):
    """
    이미지 태그 생성 함수
    """
    if not os.path.exists(file):
        return ""
    try:
        with open(file, "rb") as f:
            data = f.read()
        b64_data = base64.b64encode(data).decode()
        return f'<img src="data:image/png;base64,{b64_data}" class="{css_class}" title="{title}">'
    except:
        return ""

# 로고 이미지 로드
tag_25 = get_img_tag("25logo.png", "Team 25", css_class="top-left-logo")
tag_ajou_sw = get_img_tag("ajou_sw_logo.png", "Ajou SW", css_class="top-right-logo")
tag_ajou    = get_img_tag("ajou_logo.png", "Ajou University", css_class="top-right-logo")
tag_google  = get_img_tag("google_logo.png", "Google", css_class="top-right-logo")


# CSS 스타일링 (Eco-Intelligence 테마 적용)
st.markdown("""
<style>
    /* 폰트 설정 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', 'Helvetica Neue', sans-serif;
    }

    /* [배경색 설정] 전체 페이지: 아주 연한 웜그레이 (눈이 편안함) */
    .stApp {
        background-color: #F8F9FA; 
    }
    
    /* [상단 로고 바] 그라데이션 적용 (Eco-Tech 느낌) */
    .top-header-bar {
        background: linear-gradient(135deg, #E8F5E9 0%, #E3F2FD 100%); /* 그린에서 블루로 이어지는 은은한 그라데이션 */
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 30px;
        margin-top: -40px; /* 위쪽 여백 제거 */
        margin-bottom: 20px;
        border-bottom: 1px solid #dee2e6;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .logo-group-right {
        display: flex;
        align-items: center;
        gap: 20px;
        background-color: rgba(255, 255, 255, 0.8); /* 로고 뒤 흰색 반투명 박스 */
        padding: 8px 20px;
        border-radius: 50px; /* 둥근 모서리 */
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* 좌측 로고 (Team 25) */
    .top-left-logo {
        height: 90px; /* 적절한 크기 조절 */
        width: auto;
        object-fit: contain;
        filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.1));
    }

    /* 우측 로고들 */
    .top-right-logo {
        height: 32px;
        width: auto;
        object-fit: contain;
        transition: transform 0.3s;
    }
    .top-right-logo:hover {
        transform: scale(1.1);
    }

    /* 구분선 */
    .logo-separator {
        width: 1px;
        height: 18px;
        background-color: #ccc;
        margin: 0 5px;
    }

    /* [탭바 스타일] 모던하고 깔끔하게 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        padding-bottom: 10px;
    }
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 10px 25px !important;
        color: #555 !important;
        background-color: #FFFFFF !important;
        border-radius: 30px !important; /* 캡슐형 버튼 */
        border: 1px solid #E0E0E0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03) !important;
        transition: all 0.3s ease;
    }
    button[data-baseweb="tab"]:hover {
        background-color: #F1F8E9 !important;
        border-color: #AED581 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2E7D32 !important; /* 진한 초록색 텍스트 */
        background-color: #E8F5E9 !important; /* 연한 초록 배경 */
        border-color: #2E7D32 !important;
        box-shadow: 0 4px 6px rgba(46, 125, 50, 0.15) !important;
    }

    /* [메인 타이틀 박스] 카드 형태 디자인 */
    .header-container {
        background: white;
        padding: 40px 30px;
        border-radius: 20px;
        margin-top: 10px;
        margin-bottom: 40px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05); /* 부드러운 그림자 */
        border: 1px solid #f0f0f0;
        position: relative;
        overflow: hidden;
    }
    /* 타이틀 박스 상단에 얇은 초록색 라인 포인트 */
    .header-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 6px;
        background: linear-gradient(90deg, #66BB6A, #42A5F5);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #2c3e50; /* 진한 남색 계열 회색 */
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #7f8c8d;
        font-weight: 500;
    }

    /* Hero Section (이미지) */
    .hero-container {
        text-align: center;
        padding: 120px 20px;
        /* 배경 이미지는 유지하되 오버레이를 씌워 글씨 가독성 확보 */
        background: linear-gradient(rgba(0, 0, 50, 0.6), rgba(0, 0, 50, 0.4)), url('https://images.unsplash.com/photo-1616422285623-13ff0162193c?q=80&w=2831&auto=format&fit=crop'); 
        background-size: cover;
        background-position: center;
        border-radius: 20px;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 15px;
        text-shadow: 0 4px 8px rgba(0,0,0,0.6);
    }
    .hero-subtitle {
        font-size: 1.5rem;
        font-weight: 400;
        text-shadow: 0 2px 4px rgba(0,0,0,0.6);
        color: #f1f1f1;
    }
    
    /* 각 입력창/결과창 컨테이너 (Streamlit 기본 위젯 박스 스타일링) */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        border: 1px solid #f0f0f0;
    }

</style>
""", unsafe_allow_html=True)

# ==============================================================================
# [함수 정의] 계산 로직
# ==============================================================================
@st.cache_data
def load_real_case_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "engine1_output.csv")
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        return None

def predict_life_and_ce(decay_rate, specific_cap_base=185.0, cycles=1000):
    x = np.arange(1, cycles + 1)
    linear_fade = 0.00015 * x * decay_rate
    acc_fade = 1e-9 * np.exp(0.015 * x) * decay_rate
    cap_noise = np.random.normal(0, 0.0015, size=len(x))
    retention = 1.0 - linear_fade - acc_fade + cap_noise
    capacity = retention * specific_cap_base
    
    if decay_rate < 1.5:
        base_ce = 99.98; ce_noise_scale = 0.01
    elif decay_rate < 3.0:
        base_ce = 99.90; ce_noise_scale = 0.03
    else:
        base_ce = 99.5 - (x * 0.0005); ce_noise_scale = 0.15
        
    ce_noise = np.random.normal(0, ce_noise_scale, size=len(x))
    ce = np.clip(base_ce + ce_noise, 0, 100.0)
    return x, np.clip(capacity, 0, None), ce

def calculate_lca_impact(binder_type, solvent_type, drying_temp, loading_mass, drying_time):
    # 1. VOC
    if solvent_type == "NMP":
        voc_base = 3.0 
        voc_val = voc_base * (loading_mass / 10.0) 
        voc_desc = "Critical (NMP Toxicity)"
    else:
        voc_val = 0.0
        voc_desc = "Clean (Water Vapor)"

    # 2. CO2
    if binder_type == "PVDF":
        co2_factor = 0.45 
        chem_formula = "-(C₂H₂F₂)ₙ-"
        co2_desc = f"High ({chem_formula})"
    elif binder_type in ["CMGG", "GG", "CMC", "SBR"]:
        co2_factor = 0.12
        chem_formula = "Bio-based (C,H,O)"
        co2_desc = f"Low ({chem_formula})"
    else:
        co2_factor = 0.3
        co2_desc = "Medium"
        
    co2_val = co2_factor * (loading_mass / 20.0)

    # 3. Energy
    if solvent_type == "NMP":
        boiling_point = 204.1
        process_penalty = 1.5 
    else:
        boiling_point = 100.0
        process_penalty = 1.0

    delta_T = max(drying_temp - 25, 0)
    efficiency = 1.0 if drying_temp >= boiling_point else 0.6
    energy_val = (delta_T * drying_time * process_penalty) / (efficiency * 50000.0)
    
    return co2_val, energy_val, voc_val, co2_desc, voc_desc


# ==============================================================================
# [UI 구성] 1. 상단 로고 바 (Eco-Tech 그라데이션 적용)
# ==============================================================================
st.markdown(f"""
<div class="top-header-bar">
    <div class="logo-group-left">
        {tag_25}
    </div>
    <div class="logo-group-right">
        {tag_ajou_sw}
        {tag_ajou}
        <div class="logo-separator"></div>
        {tag_google}
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# [UI 구성] 2. 메인 네비게이션 탭
# ==============================================================================
tab_home, tab_e1, tab_e2, tab_data = st.tabs([
    "  Home  ", 
    "  Engine 1  ", 
    "  Engine 2  ", 
    "  Our Data  "
])

# 공통 헤더 HTML (탭 내부 상단 타이틀 박스 - 모던 카드 스타일)
header_html = f"""
<div class="header-container">
    <h1 class="main-title">AI 기반 배터리 소재/공정 최적화 시뮬레이터</h1>
    <div class="sub-title">Team 스물다섯 | Google-아주대학교 AI 융합 캡스톤 디자인</div>
</div>
"""

# ------------------------------------------------------------------------------
# TAB 1: Home (메인 화면)
# ------------------------------------------------------------------------------
with tab_home:
    st.markdown(header_html, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">To make the world greener <br>and sustainable</div>
        <div class="hero-subtitle">초격차 기술력을 통해 지속가능한 · 친환경 미래 사회 구현</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("### 🚀 Project Overview\n\n본 프로젝트는 **Google-아주대학교 AI 융합 캡스톤 디자인**의 일환으로 개발되었습니다. 기존의 고비용/장시간이 소요되는 배터리 소재 개발 및 공정 평가를 **AI 기반 가상 시뮬레이션**으로 대체하여 연구 효율성을 극대화하고 환경 영향을 최소화합니다.")
    with col2:
        st.success("### 💡 Key Features\n\n* **Engine 1**: AI 기반 가상 수명 예측 시뮬레이터\n* **Engine 2**: 공정 변수(LCA)에 따른 환경 영향 평가\n* **Our Data**: 실제 실험 데이터 기반 정밀 검증")

# ------------------------------------------------------------------------------
# TAB 2: Engine 1 (가상 예측)
# ------------------------------------------------------------------------------
with tab_e1:
    st.markdown(header_html, unsafe_allow_html=True)
    
    st.subheader("Engine 1. 배터리 수명 가상 시뮬레이터 (Interactive Mode)")
    st.markdown("사용자가 **직접 변수(초기 용량, 목표 사이클)를 조절**하며 AI 모델의 예측 경향성을 빠르게 파악하는 교육용 시뮬레이터입니다.")
    st.divider()
    
    col_input, col_view = st.columns([1, 2])
    with col_input:
        with st.container(border=True):
            st.markdown("#### 🔋 샘플 안정도 설정")
            sample_type = st.radio("패턴 선택", ["Perfectly Stable", "Stable", "Unstable"], label_visibility="collapsed", key="t1_radio")
            st.divider()
            st.markdown("#### ⚙️ 예측 조건 설정")
            init_cap_input = st.number_input("Initial Capacity (mAh/g)", 100.0, 400.0, 350.0)
            cycle_input = st.number_input("Prediction Cycles", 200, 2000, 500, step=50)
            run_e1 = st.button("가상 예측 실행", type="primary", use_container_width=True)

    with col_view:
        if run_e1:
            with st.spinner("AI Analyzing..."):
                if sample_type == "Perfectly Stable": decay = 0.5; label = "Perfectly Stable"; color = '#28a745'
                elif sample_type == "Stable": decay = 2.5; label = "Stable"; color = '#fd7e14'
                else: decay = 8.0; label = "Unstable"; color = '#dc3545'
                
                cycles, capacity, ce = predict_life_and_ce(decay, init_cap_input, cycle_input)
                
                fig2, (ax_cap, ax_ce) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                ax_cap.plot(cycles[:100], capacity[:100], 'k-', linewidth=2.5, label='Input Data')
                ax_cap.plot(cycles[100:], capacity[100:], '--', color=color, linewidth=2.5, label=f'Prediction ({label})')
                ax_cap.set_ylabel("Capacity (mAh/g)", fontweight='bold')
                ax_cap.set_title("Discharge Capacity Prediction", fontweight='bold')
                ax_cap.legend(); ax_cap.grid(True, alpha=0.3)
                
                ax_ce.plot(cycles, ce, '-', color='#007bff', alpha=0.8)
                ax_ce.set_ylabel("Coulombic Efficiency (%)", fontweight='bold')
                ax_ce.set_xlabel("Cycle Number", fontweight='bold')
                ax_ce.set_ylim(98.0 if decay > 5.0 else 99.5, 100.1)
                ax_ce.grid(True, alpha=0.3)
                
                st.pyplot(fig2)
                
                eol_limit = init_cap_input * 0.8
                eol_cycle = np.where(capacity < eol_limit)[0]
                if len(eol_cycle) > 0:
                    st.error(f"⚠️ **Warning:** 약 **{eol_cycle[0]} Cycle**에서 수명이 80%({eol_limit:.1f} mAh/g) 이하로 떨어집니다.")
                else:
                    st.success(f"✅ **Stable:** {cycle_input} Cycle까지 안정적입니다.")

# ------------------------------------------------------------------------------
# TAB 3: Engine 2 (공정 최적화)
# ------------------------------------------------------------------------------
with tab_e2:
    st.markdown(header_html, unsafe_allow_html=True)
    
    st.subheader("Engine 2. 공정 변수에 따른 환경 영향 예측 (LCA Optimization)")
    st.info("💡 **Update:** 본 시뮬레이터는 **화학적 조성(불소 유무)**, **용매의 독성(VOC)**, **끓는점(Boiling Point)**에 기반한 물리학적 계산 모델을 적용했습니다.")
    
    col_input_e2, col_view_e2 = st.columns([1, 2])
    
    with col_input_e2:
        with st.container(border=True):
            st.markdown("#### 🛠️ 공정 조건 설정 (음극)")
            s_binder = st.selectbox("Binder Type", ["SBR", "CMC", "CMGG", "GG", "PVDF"]) 
            s_solvent = st.radio("Solvent Type", ["Water", "NMP"])
            st.divider()
            s_temp = st.slider("Drying Temp (°C)", 60, 200, 110)
            s_time = st.slider("Drying Time (min)", 10, 720, 60) 
            s_loading = st.number_input("Loading mass (mg/cm²)", 5.0, 30.0, 10.0)
            
            st.write("")
            run_e2 = st.button("Engine 2 계산 실행", type="primary", use_container_width=True)

    with col_view_e2:
        if run_e2:
            # 1. 유효성 검사
            if s_binder == "PVDF" and s_solvent == "Water":
                st.error("🚫 **Error: 부적절한 소재 조합입니다 (Invalid Combination)**")
                st.markdown("""
                **과학적 근거 (Scientific Basis):**
                * **PVDF**는 소수성(Hydrophobic) 고분자로 물에 용해되지 않습니다.
                * PVDF를 사용하려면 반드시 **NMP**와 같은 유기 용매를 선택해야 합니다.
                """)
            elif s_binder in ["CMC", "CMGG", "GG", "SBR"] and s_solvent == "NMP":
                st.error("🚫 **Error: 부적절한 소재 조합입니다 (Invalid Combination)**")
                st.markdown(f"""
                **과학적 근거 (Scientific Basis):**
                * **{s_binder}**는 수계 바인더(Water-based Binder)로, NMP에 녹지 않습니다.
                * {s_binder}를 사용하려면 **Water** 용매를 선택해야 합니다.
                """)
            else:
                # 2. 계산 및 결과 표시
                co2, energy, voc, co2_desc, voc_desc = calculate_lca_impact(
                    s_binder, s_solvent, s_temp, s_loading, s_time
                )
                
                col1, col2, col3 = st.columns(3)
                col1.metric("CO₂ Emission", f"{co2:.4f} kg/m²", delta=co2_desc, delta_color="inverse")
                col2.metric("Energy Consumption", f"{energy:.4f} kWh/m²", help="Based on Solvent BP")
                col3.metric("VOC Emission", f"{voc:.4f} g/m²", delta=voc_desc, delta_color="inverse")
                
                st.divider()
                
                # 3. 과학적 근거 및 비교 그래프
                st.markdown("#### 📋 Scientific Basis & Comparative Analysis")
                
                with st.expander("ℹ️ 산출 근거 및 상세 분석 (Click to expand)", expanded=True):
                    st.markdown("##### 1. VOC & Solvent Toxicity")
                    if s_solvent == "NMP": st.write("🔴 **NMP (유기용매):** 높은 독성 및 VOC 발생. 배기 정화 설비 필수.")
                    else: st.write("🟢 **Water (수계용매):** 무독성, VOC 배출 없음 (수증기). 친환경 공정.")

                    st.markdown("##### 2. CO₂ & Binder Chemistry")
                    if "PVDF" in s_binder: st.write("🔴 **PVDF (불소계):** 높은 GWP(지구온난화지수), 폐기 시 환경 부담 큼.")
                    else: st.write(f"🟢 **{s_binder} (바이오/수계):** 천연 유래 소재, 낮은 탄소 발자국.")

                    st.markdown("##### 3. Process Energy (Drying)")
                    bp = 204.1 if s_solvent == "NMP" else 100
                    st.write(f"Solvent BP: **{bp}°C** vs Drying Temp: **{s_temp}°C**")
                    
                    st.divider()
                    st.markdown("##### 📊 Impact Comparison (vs NMP/PVDF Reference)")
                    
                    ref_vals = calculate_lca_impact("PVDF", "NMP", 130, s_loading, 60)[:3]
                    cur_vals = [co2, energy, voc]
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    x = np.arange(3); width = 0.35
                    rects1 = ax.bar(x - width/2, ref_vals, width, label='Ref (NMP/PVDF)', color='#FF8A80', alpha=0.7)
                    rects2 = ax.bar(x + width/2, cur_vals, width, label='Current Settings', color='#69F0AE', edgecolor='k')
                    ax.set_xticks(x); ax.set_xticklabels(['CO₂', 'Energy', 'VOC'])
                    ax.set_ylabel('Impact Value'); ax.legend(); ax.grid(axis='y', linestyle=':')
                    
                    def autolabel(rects):
                        for rect in rects:
                            h = rect.get_height()
                            ax.annotate(f'{h:.2f}', xy=(rect.get_x()+rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', fontsize=9)
                    autolabel(rects1); autolabel(rects2)
                    st.pyplot(fig)

        else:
            st.info("좌측 패널에서 공정 조건을 설정하고 [Engine 2 계산 실행]을 눌러주세요.")

# ------------------------------------------------------------------------------
# TAB 4: Our Data (실험 검증 - 맨 뒤)
# ------------------------------------------------------------------------------
with tab_data:
    st.markdown(header_html, unsafe_allow_html=True)
    
    st.subheader("Our Data. 실제 실험 데이터 검증 (Ground Truth Validation)")
    st.markdown("이 탭에서는 **Team 스물다섯이 직접 수행한 실험 데이터**를 기반으로 Engine 1의 예측 정확도를 검증합니다.")
    st.divider()

    df_results = load_real_case_data()
    if df_results is None:
        st.warning("⚠️ 'engine1_output.csv' 파일을 찾을 수 없습니다.")
    else:
        col_case_input, col_case_view = st.columns([1, 2])
        with col_case_input:
            with st.container(border=True):
                st.markdown("#### 📂 실험 케이스 선택")
                option = st.radio("데이터 선택:", ["초고속 충전 (Sample A)", "고속 충전 (Sample B)", "저속 충전 (Sample C)"], key="t2_radio")
                
                if "Sample A" in option: key = "Sample A"; st.success("✅ **Perfectly Stable** (CMGG)")
                elif "Sample B" in option: key = "Sample B"; st.warning("⚠️ **Stable** (PVDF)")
                else: key = "Sample C"; st.error("🚫 **Unstable** (Abnormal)")

        with col_case_view:
            data = df_results[df_results['Sample_Type'] == key]
            if not data.empty:
                hist = data[data['Data_Type'] == 'History']
                pred = data[data['Data_Type'] == 'Prediction']
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(hist['Cycle'], hist['Capacity'], 'o-', color='black', alpha=0.7, label='History')
                ax.plot(pred['Cycle'], pred['Capacity'], '--', color='#dc3545', linewidth=2, label='Prediction')
                ax.set_title(f"Model Validation - {key}", fontweight='bold')
                ax.set_ylabel("Capacity (mAh/g)"); ax.grid(True, alpha=0.3); ax.legend()
                st.pyplot(fig)
                
                st.info(f"📊 **AI Report**: 최종 용량 **{pred['Capacity'].iloc[-1]:.2f} mAh/g** 예측됨.")
