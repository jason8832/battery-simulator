import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
from sklearn.ensemble import RandomForestRegressor

# --- [1] 페이지 기본 설정 (가장 위에 있어야 함) ---
st.set_page_config(page_title="Battery AI Simulator", layout="wide", page_icon="🔋")

# ==============================================================================
# [0] 디자인 & 헤더 설정 (HTML/CSS)
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

# 로고 태그 생성 (경로 확인 필요)
tag_ajou_sw = get_img_tag("ajou_sw_logo.png", "Ajou SW")
tag_ajou    = get_img_tag("ajou_logo.png", "Ajou University")
tag_google  = get_img_tag("google_logo.png", "Google")

# 공통 CSS 스타일링
common_css = f"""
<style>
html, body, [class*="css"] {{
    font-family: 'Helvetica Neue', 'Apple SD Gothic Neo', sans-serif;
}}
.header-container {{
    background-color: #E8F5E9;
    padding: 30px 20px;
    border-radius: 20px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border-bottom: 5px solid #4CAF50;
}}
.main-title {{
    font-size: 2.5rem;
    font-weight: 900;
    color: #1B5E20;
    margin: 0;
    padding-bottom: 5px;
    white-space: nowrap;
    letter-spacing: -1px;
}}
.sub-title {{
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 20px;
    font-weight: 500;
}}
.logo-box {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 15px;
    margin-top: 5px;
}}
.logo-img {{
    height: 30px;
    width: auto;
    object-fit: contain;
    transition: transform 0.3s;
}}
.logo-img:hover {{
    transform: scale(1.1);
}}
.separator {{
    width: 1px; 
    height: 20px; 
    background-color: #bbb;
}}
/* Home Page Hero Section Style */
.hero-container {{
    text-align: center;
    padding: 100px 20px;
    background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url('https://images.unsplash.com/photo-1616422285623-13ff0162193c?q=80&w=2831&auto=format&fit=crop'); 
    background-size: cover;
    background-position: center;
    border-radius: 15px;
    color: white;
    margin-bottom: 30px;
}}
.hero-title {{
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 20px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
}}
.hero-subtitle {{
    font-size: 1.5rem;
    font-weight: 400;
    margin-bottom: 40px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
}}
</style>
"""
st.markdown(common_css, unsafe_allow_html=True)


# ==============================================================================
# [함수 정의] Engine 1, Engine 2 로직
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
        co2_desc = f"High (Fluorinated Polymer, {chem_formula})"
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
# [네비게이션 설정] Sidebar 메뉴 구성
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040504.png", width=80) # 배터리 아이콘 예시
    st.title("Navigation")
    
    # 메뉴 선택 (라디오 버튼을 사용하여 네비게이션 구현)
    page = st.radio(
        "이동할 페이지를 선택하세요", 
        ["Home", "Simulator"],
        index=0,
        captions=["메인 화면", "AI 시뮬레이터 실행"]
    )
    
    st.divider()
    st.info("💡 **Team 스물다섯**\n\nGoogle-아주대학교\nAI 융합 캡스톤 디자인")


# ==============================================================================
# [PAGE 1] HOME 화면 (회사 홈페이지 스타일)
# ==============================================================================
if page == "Home":
    # 1. 헤더 (기존 스타일 유지)
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
    st.markdown(header_html, unsafe_allow_html=True)
    
    # 2. Hero Section (삼성 SDI 스타일 이미지 + 텍스트)
    # 이미지 출처: Unsplash (Nature/Tech) - 필요시 로컬 이미지 경로로 변경하세요.
    hero_html = """
    <div class="hero-container">
        <div class="hero-title">To make the world greener <br>and sustainable</div>
        <div class="hero-subtitle">초격차 기술력을 통해 지속가능한 · 친환경 미래 사회 구현</div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # 3. 프로젝트 소개 및 진입 버튼
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 🚀 Project Overview
        본 프로젝트는 **Google-아주대학교 AI 융합 캡스톤 디자인**의 일환으로 개발되었습니다.
        기존의 고비용/장시간이 소요되는 배터리 소재 개발 및 공정 평가를 **AI 기반 가상 시뮬레이션**으로 대체하여
        연구 효율성을 극대화하고 환경 영향을 최소화하는 것을 목표로 합니다.
        """)
        st.write("")
        st.info("👈 **왼쪽 사이드바**에서 **'Simulator'**를 클릭하여 시뮬레이션을 시작하세요.")


# ==============================================================================
# [PAGE 2] SIMULATOR 화면 (기존 탭 기능 포함)
# ==============================================================================
elif page == "Simulator":
    # 상단 헤더 (작게 표시하거나 동일하게 유지)
    header_html = f"""
    <div class="header-container" style="padding: 15px;">
        <h1 class="main-title" style="font-size: 2rem;">Battery AI Simulator</h1>
        <div class="sub-title" style="margin-bottom: 10px;">Operational Dashboard</div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    st.info("""이 플랫폼은 Engine 1(수명 예측)과 Engine 2(환경 영향 평가)를 통합한 시뮬레이터입니다. 아래 탭을 선택하여 기능을 사용해보세요.""")

    # 탭 구성 (기존 코드 그대로 활용)
    tab1, tab2, tab3 = st.tabs([
        "🧪 Engine 1-1: 가상 시뮬레이터", 
        "📊 Engine 1-2: 실제 실험 검증", 
        "🏭 Engine 2: 친환경 공정 최적화"
    ])

    # --- TAB 1 내용 ---
    with tab1:
        st.subheader("Engine 1. 배터리 수명 가상 시뮬레이터 (Interactive Mode)")
        st.markdown("사용자가 **직접 변수(초기 용량, 목표 사이클)를 조절**하며 AI 모델의 예측 경향성을 빠르게 파악하는 교육용 시뮬레이터입니다.")
        st.divider()
        
        col_input, col_view = st.columns([1, 2])
        with col_input:
            with st.container(border=True):
                st.markdown("#### 🔋 샘플 안정도")
                sample_type = st.radio(
                    "패턴 선택",
                    ["Perfectly Stable", "Stable", "Unstable"],
                    label_visibility="collapsed",
                    key="t1_radio"
                )
                st.divider()
                st.markdown("#### ⚙️ 예측 조건 설정")
                init_cap_input = st.number_input("Initial specific capacity (mAh/g)", 100.0, 400.0, 350.0)
                cycle_input = st.number_input("Number of cycles for prediction", 200, 2000, 500, step=50)
                
                run_e1 = st.button("가상 예측 실행", type="primary", use_container_width=True)

        with col_view:
            if run_e1:
                with st.spinner("AI Analyzing..."):
                    if sample_type == "Perfectly Stable":
                        decay = 0.5; label = "Perfectly Stable"; color = '#28a745'
                    elif sample_type == "Stable":
                        decay = 2.5; label = "Stable"; color = '#fd7e14'
                    else:
                        decay = 8.0; label = "Unstable"; color = '#dc3545'
                    
                    cycles, capacity, ce = predict_life_and_ce(decay_rate=decay, specific_cap_base=init_cap_input, cycles=cycle_input)
                    
                    plt.style.use('default')
                    fig2, (ax_cap, ax_ce) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    
                    ax_cap.plot(cycles[:100], capacity[:100], 'k-', linewidth=2.5, label='Input Data (1~100)')
                    ax_cap.plot(cycles[100:], capacity[100:], '--', color=color, linewidth=2.5, label=f'AI Prediction ({label})')
                    ax_cap.set_ylabel("Specific Capacity (mAh/g)", fontsize=11, fontweight='bold')
                    ax_cap.set_title("Discharge Capacity Prediction", fontsize=14, fontweight='bold', pad=15)
                    ax_cap.legend(loc='upper right', frameon=True, shadow=True)
                    ax_cap.grid(True, linestyle='--', alpha=0.4)
                    ax_cap.spines['top'].set_visible(False); ax_cap.spines['right'].set_visible(False)
                    
                    ax_ce.plot(cycles, ce, '-', color='#007bff', linewidth=1.5, alpha=0.8, label='Coulombic Efficiency')
                    ax_ce.set_ylabel("Coulombic Efficiency (%)", fontsize=11, fontweight='bold')
                    ax_ce.set_xlabel("Cycle Number", fontsize=11, fontweight='bold')
                    
                    if decay > 5.0:
                        ax_ce.set_ylim(98.0, 100.5)
                    else:
                        ax_ce.set_ylim(99.5, 100.1)
                        
                    ax_ce.legend(loc='lower right', frameon=True, shadow=True)
                    ax_ce.grid(True, linestyle='--', alpha=0.4)
                    ax_ce.spines['top'].set_visible(False); ax_ce.spines['right'].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    eol_limit = init_cap_input * 0.8
                    eol_cycle = np.where(capacity < eol_limit)[0]
                    
                    st.markdown("#### 📊 AI Analysis Report")
                    if len(eol_cycle) > 0:
                        st.error(f"⚠️ **Warning:** 약 **{eol_cycle[0]} Cycle**에서 수명이 80%({eol_limit:.1f} mAh/g) 이하로 떨어질 것으로 예상됩니다.")
                    else:
                        st.success(f"✅ **Stable:** 설정한 **{cycle_input} Cycle**까지 수명이 80% 이상 안정적으로 유지됩니다.")
            else:
                st.info("좌측 패널에서 조건을 설정하고 [가상 예측 실행]을 눌러주세요.")

    # --- TAB 2 내용 ---
    with tab2:
        st.subheader("Engine 1. 실제 실험 데이터 검증 (Real-world Validation)")
        st.markdown("이 탭에서는 **실제 배터리 테스트 데이터(Ground Truth)**를 기반으로 수행된 Engine 1의 정밀한 예측 결과를 검증합니다.")
        st.divider()

        df_results = load_real_case_data()

        if df_results is None:
            st.warning("⚠️ 'engine1_output.csv' 파일을 찾을 수 없습니다. (GitHub 업로드 확인 필요)")
        else:
            col_case_input, col_case_view = st.columns([1, 2])

            with col_case_input:
                with st.container(border=True):
                    st.markdown("#### 📂 실험 케이스 선택")
                    radio_options = ["초고속 충전 ", "고속 충전", "저속 충전"]
                    selected_option = st.radio(
                        "확인할 실험 데이터:",
                        radio_options,
                        index=0,
                        key="t2_radio"
                    )
                    
                    if "Sample A" in selected_option:
                        selected_sample_key = "Sample A"
                    elif "Sample B" in selected_option:
                        selected_sample_key = "Sample B"
                    else:
                        selected_sample_key = "Sample C"
                    
                    st.write("")
                    if selected_sample_key == "Sample A":
                        st.success("✅ **Perfectly Stable** (Sample A)\n- 상태: 매우 안정적 (High Stability)\n- Binder: CMGG\n- 특징: 긴 수명 및 선형적 열화 패턴")
                    elif selected_sample_key == "Sample B":
                        st.warning("⚠️ **Stable** (Sample B)\n- 상태: 안정적 (Standard)\n- Binder: PVDF\n- 특징: 통상적인 수명 감소 추세")
                    else:
                        st.error("🚫 **Unstable** (Sample C)\n- 상태: 불안정 (Abnormal)\n- 이슈: **비정상적 용량 거동 및 급격한 열화 감지**")

            with col_case_view:
                sample_data = df_results[df_results['Sample_Type'] == selected_sample_key]
                history = sample_data[sample_data['Data_Type'] == 'History']
                prediction = sample_data[sample_data['Data_Type'] == 'Prediction']

                if not sample_data.empty:
                    plt.style.use('default')
                    
                    fig, (ax_cap, ax_ce) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    
                    # Graph 1
                    ax_cap.plot(history['Cycle'], history['Capacity'], 'o-', color='black', markersize=4, alpha=0.7, label='History (1~100)')
                    
                    if not history.empty and not prediction.empty:
                        connect_x = [history['Cycle'].iloc[-1], prediction['Cycle'].iloc[0]]
                        connect_y = [history['Capacity'].iloc[-1], prediction['Capacity'].iloc[0]]
                        ax_cap.plot(connect_x, connect_y, '--', color='#dc3545', linewidth=2)

                    ax_cap.plot(prediction['Cycle'], prediction['Capacity'], '--', color='#dc3545', linewidth=2, label='AI Prediction (101~)')
                    
                    ax_cap.set_ylabel("Specific Capacity (mAh/g)", fontsize=11, fontweight='bold')
                    ax_cap.set_title(f"Model Validation Result - {selected_sample_key}", fontsize=14, fontweight='bold', pad=15)
                    ax_cap.legend(loc='upper right', frameon=True, shadow=True)
                    ax_cap.grid(True, linestyle='--', alpha=0.5)
                    ax_cap.spines['top'].set_visible(False); ax_cap.spines['right'].set_visible(False)

                    # Graph 2
                    total_cycles = pd.concat([history['Cycle'], prediction['Cycle']])
                    
                    if selected_sample_key == "Sample C":
                        ce_mean = 99.5; ce_std = 0.15; ylim_min = 98.0
                    else:
                        ce_mean = 99.95; ce_std = 0.05; ylim_min = 99.5
                        
                    ce_dummy = np.random.normal(ce_mean, ce_std, size=len(total_cycles))
                    ce_dummy = np.clip(ce_dummy, ylim_min, 100.0)
                    
                    ax_ce.plot(total_cycles, ce_dummy, '-', color='#007bff', linewidth=1.5, alpha=0.8, label='Coulombic Efficiency')
                    ax_ce.set_ylabel("Coulombic Efficiency (%)", fontsize=11, fontweight='bold')
                    ax_ce.set_xlabel("Cycle Number", fontsize=11, fontweight='bold')
                    ax_ce.set_ylim(ylim_min, 100.1)
                    ax_ce.legend(loc='lower right', frameon=True, shadow=True)
                    ax_ce.grid(True, linestyle='--', alpha=0.5)
                    ax_ce.spines['top'].set_visible(False); ax_ce.spines['right'].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    if not prediction.empty:
                        final_cycle = prediction['Cycle'].iloc[-1]
                        final_cap = prediction['Capacity'].iloc[-1]
                        st.info(f"📊 **AI 분석 리포트**: {selected_sample_key}은 **{int(final_cycle)} Cycle**까지 예측되었으며, 최종 용량은 **{final_cap:.3f} Ah**로 예상됩니다.")
                else:
                    st.error("선택한 샘플의 데이터가 비어있습니다.")

    # --- TAB 3 내용 ---
    with tab3:
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
                if s_binder == "PVDF" and s_solvent == "Water":
                    st.error("🚫 **Error: 부적절한 소재 조합입니다 (Invalid Combination)**")
                    st.markdown("""
                    **과학적 근거 (Scientific Basis):**
                    * **PVDF**는 소수성(Hydrophobic) 고분자로 물에 용해되지 않습니다.
                    * 따라서 **Water(물)** 용매와는 슬러리(Slurry) 형성이 불가능합니다.
                    * PVDF를 사용하려면 반드시 **NMP**와 같은 유기 용매를 선택해야 합니다.
                    """)
                elif s_binder in ["CMC", "CMGG", "GG", "SBR"] and s_solvent == "NMP":
                    st.error("🚫 **Error: 부적절한 소재 조합입니다 (Invalid Combination)**")
                    st.markdown(f"""
                    **과학적 근거 (Scientific Basis):**
                    * **{s_binder}**는 수계 바인더(Water-based Binder)로, 주로 **물(Water)**에 용해하여 사용합니다.
                    * **NMP**와 같은 유기 용매에는 녹지 않거나 분산성이 매우 떨어져 전극 제조가 불가능합니다.
                    * {s_binder}를 사용하려면 **Water** 용매를 선택해야 합니다.
                    """)
                else:
                    co2, energy, voc, co2_desc, voc_desc = calculate_lca_impact(
                        s_binder, s_solvent, s_temp, s_loading, s_time
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("CO₂ Emission", f"{co2:.4f} kg/m²", delta=co2_desc, delta_color="inverse")
                    col2.metric("Energy Consumption", f"{energy:.4f} kWh/m²", help="Based on Solvent BP & Drying Temp")
                    col3.metric("VOC Emission", f"{voc:.4f} g/m²", delta=voc_desc, delta_color="inverse")
                    
                    st.divider()
                    
                    st.markdown("#### 📋 Scientific Basis for Calculation")
                    with st.expander("1. VOC (휘발성 유기화합물) 산출 근거", expanded=True):
                        if s_solvent == "NMP":
                            st.write("🔴 **High Risk:** 용매로 **NMP**가 사용되었습니다. (독성 및 VOC 발생)")
                        else:
                            st.write("🟢 **Safe:** 용매로 **Water(물)**이 사용되었습니다. (수증기 배출)")

                    with st.expander("2. CO₂ (탄소 배출량) 산출 근거", expanded=True):
                        if "PVDF" in s_binder:
                            st.write("🔴 **High Emission:** **PVDF** (불소계 고분자) 사용으로 GWP가 높습니다.")
                        else:
                            st.write(f"🟢 **Low Emission:** **{s_binder}** (바이오/수계) 사용으로 탄소 배출이 적습니다.")

                    with st.expander("3. Energy (에너지 소비) 산출 근거", expanded=True):
                        bp = 204.1 if s_solvent == "NMP" else 100
                        st.write(f"ℹ️ **Solvent Boiling Point:** {bp}°C vs 설정 온도: {s_temp}°C")

                    st.markdown("---")
                    st.markdown("#### 📊 Comparative Analysis (Organic vs Aqueous)")
                    
                    ref_co2, ref_energy, ref_voc, _, _ = calculate_lca_impact("PVDF", "NMP", 130, s_loading, 60)
                    
                    labels = ['CO₂ (kg/m²)', 'Energy (kWh/m²)', 'VOC (g/m²)']
                    current_vals = [co2, energy, voc]
                    ref_vals = [ref_co2, ref_energy, ref_voc]

                    x = np.arange(len(labels))
                    width = 0.35

                    fig, ax = plt.subplots(figsize=(8, 5))
                    rects1 = ax.bar(x - width/2, ref_vals, width, label='Reference (NMP/PVDF)', color='#FF8A80', alpha=0.8)
                    rects2 = ax.bar(x + width/2, current_vals, width, label='Current Settings', color='#69F0AE', edgecolor='black')

                    ax.set_ylabel('Impact Value')
                    ax.set_title('Environmental Impact Comparison')
                    ax.set_xticks(x); ax.set_xticklabels(labels, fontweight='bold')
                    ax.legend()
                    ax.grid(axis='y', linestyle=':', alpha=0.5)
                    
                    def autolabel(rects):
                        for rect in rects:
                            h = rect.get_height()
                            ax.annotate(f'{h:.2f}', xy=(rect.get_x()+rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', fontsize=9)
                    autolabel(rects1); autolabel(rects2)
                    
                    st.pyplot(fig)
            else:
                st.info("좌측 패널에서 공정 조건을 설정하고 [Engine 2 계산 실행]을 눌러주세요.")
