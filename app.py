import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
from sklearn.ensemble import RandomForestRegressor

# --- [1] 페이지 기본 설정 ---
st.set_page_config(page_title="Battery AI Simulator", layout="wide", page_icon="🔋")

# ==============================================================================
# [0] 디자인 & CSS 설정
# ==============================================================================

def get_img_tag(file, title):
    if not os.path.exists(file):
        return ""
    try:
        with open(file, "rb") as f:
            data = f.read()
        b64_data = base64.b64encode(data).decode()
        return f'<img src="data:image/png;base64,{b64_data}" class="logo-img" title="{title}">'
    except:
        return ""

# 로고 태그 생성
tag_ajou_sw = get_img_tag("ajou_sw_logo.png", "Ajou SW")
tag_ajou    = get_img_tag("ajou_logo.png", "Ajou University")
tag_google  = get_img_tag("google_logo.png", "Google")

# CSS 스타일링 (탭 위치 조정 및 디자인)
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', 'Apple SD Gothic Neo', sans-serif;
    }
    
    /* 메인 화면 상단 여백 줄이기 (탭을 더 위로) */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* 헤더 컨테이너 스타일 */
    .header-container {
        background-color: #E8F5E9;
        padding: 20px 20px;
        border-radius: 15px;
        margin-top: 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-bottom: 4px solid #4CAF50;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1B5E20;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #555;
        margin-top: 5px;
        margin-bottom: 15px;
        font-weight: 500;
    }
    .logo-box {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
    }
    .logo-img {
        height: 28px;
        width: auto;
        object-fit: contain;
        transition: transform 0.3s;
    }
    .logo-img:hover {
        transform: scale(1.1);
    }
    .separator {
        width: 1px; height: 18px; background-color: #bbb;
    }
    
    /* 탭 버튼 스타일 커스텀 */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 700 !important;
        padding: 0px 20px !important;
    }
    
    /* Hero Section (Home) */
    .hero-container {
        text-align: center;
        padding: 80px 20px;
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url('https://images.unsplash.com/photo-1616422285623-13ff0162193c?q=80&w=2831&auto=format&fit=crop'); 
        background-size: cover;
        background-position: center;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
    }
    .hero-title {
        font-size: 3.0rem;
        font-weight: 800;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 400;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# [함수 정의] Engine 로직
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
    if solvent_type == "NMP":
        voc_base = 3.0; voc_val = voc_base * (loading_mass / 10.0); voc_desc = "Critical (NMP Toxicity)"
    else:
        voc_val = 0.0; voc_desc = "Clean (Water Vapor)"

    if binder_type == "PVDF":
        co2_factor = 0.45; chem_formula = "-(C₂H₂F₂)ₙ-"
        co2_desc = f"High ({chem_formula})"
    elif binder_type in ["CMGG", "GG", "CMC", "SBR"]:
        co2_factor = 0.12; chem_formula = "Bio-based"
        co2_desc = f"Low ({chem_formula})"
    else:
        co2_factor = 0.3; co2_desc = "Medium"
    co2_val = co2_factor * (loading_mass / 20.0)

    bp = 204.1 if solvent_type == "NMP" else 100.0
    process_penalty = 1.5 if solvent_type == "NMP" else 1.0
    delta_T = max(drying_temp - 25, 0)
    efficiency = 1.0 if drying_temp >= bp else 0.6
    energy_val = (delta_T * drying_time * process_penalty) / (efficiency * 50000.0)
    
    return co2_val, energy_val, voc_val, co2_desc, voc_desc


# ==============================================================================
# [UI 구성] 1. 메인 네비게이션 탭 (최상단 배치)
# ==============================================================================
# [수정됨] 탭을 가장 먼저 선언하여 화면 최상단에 위치시킴
tab_home, tab_e1, tab_e2, tab_data = st.tabs([
    "🏠 Home", 
    "🧪 Engine 1: 가상 예측", 
    "🏭 Engine 2: 공정 최적화",
    "📂 Our Data: 실험 검증"
])

# 공통으로 사용할 헤더 HTML (모든 탭 안에 삽입됨)
header_html = f"""
<div class="header-container">
    <h1 class="main-title">AI 기반 배터리 소재/공정 최적화 시뮬레이터</h1>
    <div class="sub-title">Team 스물다섯 | Google-아주대학교 AI 융합 캡스톤 디자인</div>
    <div class="logo-box">
        {tag_ajou_sw}
        {tag_ajou}
        <div class="separator"></div>
        {tag_google}
    </div>
</div>
"""

# ------------------------------------------------------------------------------
# TAB 1: HOME (메인 화면)
# ------------------------------------------------------------------------------
with tab_home:
    st.markdown(header_html, unsafe_allow_html=True) # 헤더 삽입
    
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">To make the world greener <br>and sustainable</div>
        <div class="hero-subtitle">초격차 기술력을 통해 지속가능한 · 친환경 미래 사회 구현</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("### 🚀 Project Overview\n\n본 프로젝트는 **Google-아주대학교 AI 융합 캡스톤 디자인**의 일환으로 개발되었습니다. 기존의 고비용/장시간이 소요되는 배터리 소재 개발 및 공정 평가를 **AI 기반 가상 시뮬레이션**으로 대체하여 연구 효율성을 극대화합니다.")
    with col2:
        st.success("### 💡 Key Features\n\n* **Engine 1**: AI 기반 가상 수명 예측 시뮬레이터\n* **Engine 2**: 공정 변수(LCA)에 따른 환경 영향 평가\n* **Our Data**: 실제 실험 데이터 기반 정밀 검증")

# ------------------------------------------------------------------------------
# TAB 2: Engine 1 (가상 시뮬레이터)
# ------------------------------------------------------------------------------
with tab_e1:
    st.markdown(header_html, unsafe_allow_html=True) # 헤더 삽입
    
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

# ------------------------------------------------------------------------------
# TAB 3: Engine 2 (친환경 공정 최적화)
# ------------------------------------------------------------------------------
with tab_e2:
    st.markdown(header_html, unsafe_allow_html=True) # 헤더 삽입
    
    st.subheader("Engine 2. 공정 변수에 따른 환경 영향 예측 (LCA Optimization)")
    st.info("💡 **Update:** 화학적 조성(불소 유무), 용매 독성(VOC), 끓는점(Energy)에 기반한 물리학적 계산 모델입니다.")
    
    col_in, col_out = st.columns([1, 2])
    with col_in:
        with st.container(border=True):
            st.markdown("#### 🛠️ 공정 조건 설정")
            binder = st.selectbox("Binder", ["SBR", "CMC", "CMGG", "GG", "PVDF"])
            solvent = st.radio("Solvent", ["Water", "NMP"])
            st.divider()
            temp = st.slider("Temp (°C)", 60, 200, 110)
            time = st.slider("Time (min)", 10, 720, 60)
            load = st.number_input("Loading (mg/cm²)", 5.0, 30.0, 10.0)
            run_e2 = st.button("계산 실행", type="primary", use_container_width=True)

    with col_out:
        if run_e2:
            if binder == "PVDF" and solvent == "Water":
                st.error("🚫 **PVDF는 물에 녹지 않습니다.** (NMP 필요)")
            elif binder in ["CMC", "CMGG", "GG", "SBR"] and solvent == "NMP":
                st.error(f"🚫 **{binder}는 수계 바인더입니다.** (Water 필요)")
            else:
                co2, eng, voc, d_co2, d_voc = calculate_lca_impact(binder, solvent, temp, load, time)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("CO₂ Emission", f"{co2:.3f}", delta=d_co2, delta_color="inverse")
                c2.metric("Energy", f"{eng:.3f}", help="kWh/m²")
                c3.metric("VOCs", f"{voc:.3f}", delta=d_voc, delta_color="inverse")
                
                st.divider()
                st.markdown("#### 📊 Comparative Analysis")
                ref_vals = calculate_lca_impact("PVDF", "NMP", 130, load, 60)[:3]
                cur_vals = [co2, eng, voc]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                x = np.arange(3); width = 0.35
                ax.bar(x - width/2, ref_vals, width, label='Ref (PVDF/NMP)', color='#FF8A80')
                ax.bar(x + width/2, cur_vals, width, label='Current', color='#69F0AE', edgecolor='k')
                ax.set_xticks(x); ax.set_xticklabels(['CO₂', 'Energy', 'VOC'])
                ax.legend(); ax.grid(axis='y', linestyle=':')
                st.pyplot(fig)

# ------------------------------------------------------------------------------
# TAB 4: Our Data (실제 실험 검증 - 맨 뒤)
# ------------------------------------------------------------------------------
with tab_data:
    st.markdown(header_html, unsafe_allow_html=True) # 헤더 삽입
    
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
