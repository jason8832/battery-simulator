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
# [사용자 설정] 팀원 정보 편집
# ==============================================================================
team_members = [
    {
        "name": "이하영",
        "role": "TEAM LEADER",
        "desc": "프로젝트 총괄 기획 및 웹사이트 UI/UX 디자인",
        "tags": ["#PM", "#Design"],
        "photo_file": "Profile1.jpeg"
    },
    {
        "name": "정회권",
        "role": "DEVELOPER",
        "desc": "시뮬레이션 알고리즘 설계·코딩 및 웹사이트 구현",
        "tags": ["#Algorithm", "#Web_Dev"],
        "photo_file": "Profile3.jpeg"
    },
    {
        "name": "신동하",
        "role": "DATA ANALYST",
        "desc": "배터리 실험 결과 해석 및 시뮬레이션 데이터 분석",
        "tags": ["#Data_Analysis", "#Insight"],
        "photo_file": "Profile5.jpeg"
    },
    {
        "name": "권현정",
        "role": "CHEMICAL RESEARCHER",
        "desc": "친환경 소재 바인더 화학적 검증 및 배터리 성능 실험",
        "tags": ["#Chemistry", "#Experiment"],
        "photo_file": "Profile6.jpeg"
    },
    {
        "name": "박재찬",
        "role": "RESEARCHER & ANALYST",
        "desc": "배터리 실험 수행 및 시뮬레이션 데이터 분석",
        "tags": ["#Experiment", "#Data_Analysis"],
        "photo_file": "Profile2.jpeg"
    }
]

# ==============================================================================
# [0] 디자인 & CSS 설정
# ==============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

def get_base64_image(filename):
    """이미지 파일을 읽어 Base64 문자열로 반환"""
    if not filename: return None
    file_path = os.path.join(current_dir, filename)
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

def get_img_tag(filename, title, css_class="logo-img"):
    """HTML <img> 태그 생성"""
    b64 = get_base64_image(filename)
    if b64:
        return f'<img src="data:image/png;base64,{b64}" class="{css_class}" title="{title}">'
    return ""

# 1. 이미지 자원 로드
tag_25 = get_img_tag("25logo.png", "Team 25", css_class="top-left-logo")
tag_ajou_sw = get_img_tag("ajou_sw_logo.png", "Ajou SW", css_class="top-right-logo")
tag_ajou    = get_img_tag("ajou_logo.png", "Ajou University", css_class="top-right-logo")
tag_google  = get_img_tag("google_logo.png", "Google", css_class="top-right-logo")

# 2. 상단 배경 설정
header_bg_style = "background-color: #B1B6B0;"
    
# ------------------------------------------------------------------------------
# 3. CSS 스타일링
# ------------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Noto Sans KR', 'Helvetica Neue', sans-serif;
    }}

    /* 전체 배경색: 차분한 세이지 그레이 */
    .stApp {{
        background-color: #DAE0DD; 
    }}
    
    /* [수정됨] 조건 설정 네모칸(Border Wrapper) 강력 스타일링 */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: #B1B6B0 !important;  /* 요청하신 배경색 적용 */
        border: 5px solid #1B5E20 !important;  /* 테두리 굵기 5px */
        border-radius: 15px !important;       
        padding: 20px !important;              
        box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
    }}
    
    /* 내부 요소가 배경을 가리지 않도록 투명 처리 */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        background-color: transparent !important; 
    }}

    /* 상단 로고 바 */
    .top-header-bar {{
        {header_bg_style}
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 25px;
        margin-top: -30px;
        margin-bottom: 20px;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-bottom: 3px solid #2E7D32;
    }}
    
    .logo-group-right {{
        display: flex;
        align-items: center;
        gap: 20px;
        background-color: rgba(255, 255, 255, 0.7);
        padding: 5px 15px;
        border-radius: 10px;
        border: none !important; /* [수정됨] 테두리 제거 */
    }}

    .top-left-logo {{ height: 120px; width: auto; object-fit: contain; filter: drop-shadow(2px 2px 2px rgba(0,0,0,0.3)); }}
    .top-right-logo {{ height: 35px; width: auto; object-fit: contain; transition: transform 0.3s; }}
    .top-right-logo:hover {{ transform: scale(1.1); }}
    .logo-separator {{ width: 2px; height: 20px; background-color: #333; margin: 0 5px; }}

    /* 탭바 스타일 */
    button[data-baseweb="tab"] {{
        font-size: 18px !important;
        font-weight: 800 !important;
        padding: 10px 30px !important;
        color: #333 !important;
        background-color: rgba(255,255,255,0.6) !important;
        margin: 0 5px !important;
        border-radius: 10px 10px 0 0 !important;
        border: 2px solid #2E7D32 !important;
        border-bottom: none !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #d32f2f !important;
        background-color: #ffffff !important;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1) !important;
    }}

    /* 헤더 컨테이너 */
    @keyframes gradientAnimation {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .header-container {{
        background: linear-gradient(-45deg, #E8F5E9, #C8E6C9, #B2DFDB, #E0F2F1, #FFFFFF);
        background-size: 400% 400%;
        animation: gradientAnimation 8s ease infinite;
        padding: 40px 30px;
        border-radius: 15px;
        margin-top: 10px;
        margin-bottom: 30px;
        text-align: center;
        border: 3px solid #2E7D32; 
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }}
    .main-title {{
        font-size: 2.8rem;
        font-weight: 900;
        color: #1B5E20;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 1px 1px 0px rgba(255,255,255,0.8);
    }}
    .sub-title {{
        font-size: 1.1rem;
        color: #333;
        margin-top: 10px;
        font-weight: 600;
    }}
    
    /* Hero Section */
    .hero-container {{
        text-align: center;
        padding: 100px 20px;
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url('https://images.unsplash.com/photo-1616422285623-13ff0162193c?q=80&w=2831&auto=format&fit=crop'); 
        background-size: cover;
        background-position: center;
        border-radius: 20px;
        color: white;
        margin-bottom: 40px;
        border: 3px solid #2E7D32;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }}
    .hero-title {{
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 20px;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.8);
    }}
    .hero-subtitle {{
        font-size: 1.5rem;
        font-weight: 400;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
    }}

    /* 페르소나 카드 스타일 */
    .persona-card {{
        display: flex;
        flex-direction: row; 
        align-items: center;
        background-color: white;
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        min-height: 140px;
    }}
    .persona-card:hover {{
        transform: translateY(-3px);
        border-color: #2E7D32;
        box-shadow: 0 8px 16px rgba(46, 125, 50, 0.15);
    }}
    .persona-img {{
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 20px;
        border: 2px solid #E8F5E9;
        background-color: #F1F8E9;
        flex-shrink: 0;
    }}
    .persona-content {{ text-align: left; width: 100%; }}
    .persona-name {{ font-size: 1.3rem; font-weight: 800; color: #1B5E20; margin-bottom: 4px; }}
    .persona-role {{ font-size: 0.85rem; color: #555; font-weight: 700; margin-bottom: 8px; text-transform: uppercase; background-color: #E8F5E9; padding: 2px 8px; border-radius: 4px; display: inline-block; }}
    .persona-desc {{ font-size: 0.95rem; color: #333; line-height: 1.5; margin-bottom: 12px; }}
    .tag-badge {{ background-color: #E3F2FD; color: #1565C0; padding: 4px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; margin-right: 5px; display: inline-block; margin-top: 2px; }}
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
        
        # 데이터의 앞뒤 공백을 제거하여 매칭 오류 방지
        if 'Sample_Type' in df.columns:
            df['Sample_Type'] = df['Sample_Type'].astype(str).str.strip()
            
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
        co2_factor = 0.45; chem_formula = "-(C₂H₂F₂)ₙ-"; co2_desc = f"High ({chem_formula})"
    elif binder_type in ["CMGG", "GG", "CMC"]: 
        co2_factor = 0.12; chem_formula = "Bio-based (C,H,O)"; co2_desc = f"Low ({chem_formula})"
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
# [UI 구성] 1. 상단 로고 바
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
    "  Home  ", 
    "  Engine 1  ", 
    "  Engine 2  ", 
    "  Our Data  "
])

# 대제목 헤더 박스
header_html = f"""
<div class="header-container">
    <h1 class="main-title">AI 기반 배터리 성능·환경 영향 시뮬레이터</h1>
    <div class="sub-title">Team 스물다섯 | Google-아주대학교 AI 융합 캡스톤 디자인 프로젝트</div>
</div>
"""

# ------------------------------------------------------------------------------
# TAB 1: Home
# ------------------------------------------------------------------------------
with tab_home:
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">To make the world greener <br>and sustainable</div>
        <div class="hero-subtitle">초격차 기술력을 통해 지속가능한 · 친환경 미래 사회 구현</div>
    </div>
    """, unsafe_allow_html=True)

    # Project Overview & Key Features
    col1, col2 = st.columns([1, 1])
    with col1:
       st.info("""### 🚀 Project Overview
 본 프로젝트는 아주대학교 화학공학과 캡스톤 디자인에서 시작되어, Google-아주대학교 융합 캡스톤 디자인의 일환으로 만들어졌습니다.
 이 웹페이지는 배터리가 얼마나 오래 사용할 수 있는지, 시간이 지나도 성능이 얼마나 유지되는지를 예측하고, 공정 조건에 따라 에너지 사용량과 환경 부담이 어떻게 달라지는지를 가상 실험으로 살펴볼 수 있는 도구입니다.
""")
    with col2:
        st.success("### 💡 Key Features\n\n* **Engine 1**: 배터리 성능 예측 시뮬레이터\n* **Engine 2**: 공정 환경 영향 시뮬레이터\n* **Our Data**: 실제 실험 데이터 검증 ")

    st.markdown("---")
    
    # [Team Member Section]
    st.markdown("<h3 style='color: #1B5E20; margin-bottom: 20px;'> Group Member 👥 </h3>", unsafe_allow_html=True)
    
    cols = st.columns(2) 
    
    for i, member in enumerate(team_members):
        col_idx = i % 2
        tags_html = "".join([f'<span class="tag-badge">{tag}</span>' for tag in member['tags']])
        
        # 파일명으로 이미지 찾기
        profile_b64 = get_base64_image(member["photo_file"])
        
        # 이미지가 있으면 로컬 사진, 없으면 기본 아바타 (Fallback)
        if profile_b64:
            img_src = f"data:image/jpeg;base64,{profile_b64}"
        else:
            img_src = f"https://api.dicebear.com/7.x/avataaars/svg?seed={member['name']}"

        with cols[col_idx]:
            st.markdown(f"""
            <div class="persona-card">
                <img src="{img_src}" class="persona-img">
                <div class="persona-content">
                    <div class="persona-name">{member['name']}</div>
                    <div class="persona-role">{member['role']}</div>
                    <div class="persona-desc">{member['desc']}</div>
                    <div class="persona-tags">{tags_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # [수정] 하단 푸터 로고 (Bottom Right Footer Logo) - 공과대학 로고만 표시
    # ==========================================================================
    st.write("")  # 여백 추가
    st.write("")
    
    # 파일명 정의
    file_eng = "01_(국영문)공과대학.png"
    
    # Base64 변환
    b64_eng = get_base64_image(file_eng)

    if b64_eng:
        html_content = f"""
        <div style="
            display: flex; 
            justify-content: flex-end;    /* 우측 정렬 */
            align-items: center; 
            margin-top: 80px;             /* 위쪽 요소와의 간격 */
            margin-bottom: 40px; 
            padding-right: 10px;">
            <img src="data:image/png;base64,{b64_eng}" 
                 style="width: 320px; max-width: 100%; opacity: 0.9; filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.1));">
        </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: Engine 1
# ------------------------------------------------------------------------------
with tab_e1:
    st.markdown(header_html, unsafe_allow_html=True)
    
    st.subheader("Engine 1. 배터리 성능 예측 시뮬레이터 ")
    st.markdown("사용자가 직접 변수(초기 용량, 목표 사이클)를 조절하며 AI 모델의 예측 경향성을 빠르게 파악하는 시뮬레이터입니다.")
    st.divider()
    
    col_input, col_view = st.columns([1, 2])
    with col_input:
        # [확인용] CSS에서 div[data-testid="stVerticalBlockBorderWrapper"]를 강제로 스타일링 중입니다.
        with st.container(border=True): 
            st.markdown("#### 🔋 충/방전 속도")
            # [수정됨] Engine 1 선택 목록을 속도별(Slow/Charge/Fast)로 유지
            sample_type = st.radio("패턴 선택", ["Slow Charge/Discharge", "Charge/Discharge", "Fast Charge/Discharge"], label_visibility="collapsed", key="t1_radio")
            st.divider()
            st.markdown("#### ⚙️ 예측 조건 설정")
            init_cap_input = st.number_input("Initial specific capacity (mAh/g)", 100.0, 400.0, 350.0)
            cycle_input = st.number_input("Number of cycles for prediction", 200, 2000, 500, step=50)
            run_e1 = st.button("가상 예측 실행", type="primary", use_container_width=True)

    with col_view:
        if run_e1:
            with st.spinner("AI Analyzing..."):
                # [수정됨] 그래프 라벨도 선택한 속도명과 일치시킴
                if sample_type == "Slow Charge/Discharge": decay = 0.5; label = "Perfectly Stable"; color = '#28a745'
                elif sample_type == "Charge/Discharge": decay = 2.5; label = "Stable"; color = '#fd7e14'
                else: decay = 8.0; label = "Unstable"; color = '#dc3545'
                
                cycles, capacity, ce = predict_life_and_ce(decay, init_cap_input, cycle_input)
                
                fig2, (ax_cap, ax_ce) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                ax_cap.plot(cycles[:100], capacity[:100], 'k-', linewidth=2.5, label='Input Data')
                ax_cap.plot(cycles[100:], capacity[100:], '--', color=color, linewidth=2.5, label=f'Prediction ({label})')
                ax_cap.set_ylabel("Specific Capacity (mAh/g)", fontweight='bold')
                ax_cap.set_title("Performance Prediction", fontweight='bold')
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
# TAB 3: Engine 2 
# ------------------------------------------------------------------------------
with tab_e2:
    st.markdown(header_html, unsafe_allow_html=True)
    
    st.subheader("Engine 2. 공정 환경 영향 시뮬레이터 ")
    st.info(" 본 시뮬레이터는 화학적 조성(불소 유무), 용매의 독성(VOC), 끓는점(Boiling Point)에 기반한 물리학적 계산 모델을 적용했습니다.")
    
    col_input_e2, col_view_e2 = st.columns([1, 2])
    
    with col_input_e2:
        with st.container(border=True): 
            st.markdown("#### 🛠️ 공정 조건 설정 ")
            s_binder = st.selectbox("Binder Type", ["CMC", "CMGG", "GG", "PVDF"]) 
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
                * PVDF를 사용하려면 반드시 **NMP**와 같은 유기 용매를 선택해야 합니다.
                """)
            elif s_binder in ["CMC", "CMGG", "GG"] and s_solvent == "NMP":
                st.error("🚫 **Error: 부적절한 소재 조합입니다 (Invalid Combination)**")
                st.markdown(f"""
                **과학적 근거 (Scientific Basis):**
                * **{s_binder}**는 수계 바인더(Water-based Binder)로, NMP에 녹지 않습니다.
                * {s_binder}를 사용하려면 **Water** 용매를 선택해야 합니다.
                """)
            else:
                co2, energy, voc, co2_desc, voc_desc = calculate_lca_impact(
                    s_binder, s_solvent, s_temp, s_loading, s_time
                )
                
                col1, col2, col3 = st.columns(3)
                col1.metric("CO₂ Emission", f"{co2:.4f} kg/m²", delta=co2_desc, delta_color="inverse")
                col2.metric("Energy Consumption", f"{energy:.4f} kWh/m²", help="Based on Solvent BP")
                col3.metric("VOC Emission", f"{voc:.4f} g/m²", delta=voc_desc, delta_color="inverse")
                
                st.divider()
                
                # [수정] 아래 섹션도 왼쪽 설정 박스와 동일한 스타일 적용 (배경색 및 테두리)
                with st.container(border=True):
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
# TAB 4: Our Data
# ------------------------------------------------------------------------------
with tab_data:
    st.markdown(header_html, unsafe_allow_html=True)
    
    st.subheader("Our Data. 실제 실험 데이터 검증 ")
    st.markdown("  직접 수행한 실험 데이터*를 기반으로 Engine 1 Mechanism의 예측 정확도를 검증합니다.")
    st.divider()

    df_results = load_real_case_data()
    if df_results is None:
        st.warning("⚠️ 'engine1_output.csv' 파일을 찾을 수 없습니다.")
    else:
        col_case_input, col_case_view = st.columns([1, 2])
        with col_case_input:
            with st.container(border=True): 
                st.markdown("#### 🔋 충/방전 속도")
                # [수정됨] 괄호 내용 삭제 (Sample A/B/C)
                option = st.radio("데이터 선택:", ["Slow Charge/Discharge", "Charge/Discharge", "Fast Charge/Discharge"], key="t2_radio")
                
                # [수정됨] 안내문구에서 괄호 삭제 (CMGG, PVDF 등)
                if option == "Slow Charge/Discharge":
                    csv_key = "Slow Charge/Discharge"
                    st.success("✅ **Perfectly Stable**")
                elif option == "Charge/Discharge":
                    csv_key = "Charge/Discharge"
                    st.warning("⚠️ **Stable**")
                else: 
                    csv_key = "Fast Charge/Discharge"
                    st.error("🚫 **Unstable**")

        with col_case_view:
            # 매핑된 csv_key로 필터링 (공백 제거된 상태에서 매칭)
            data = df_results[df_results['Sample_Type'] == csv_key]
            
            if not data.empty:
                hist = data[data['Data_Type'] == 'History']
                pred = data[data['Data_Type'] == 'Prediction']
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # [수정됨] History: 점 그래프 (원형)
                ax.scatter(hist['Cycle'], hist['Capacity'], color='black', alpha=0.6, s=25, label='History')
                
                # [수정됨] Prediction: 점 그래프 (사각형) - 요청 반영
                ax.scatter(pred['Cycle'], pred['Capacity'], color='#dc3545', alpha=0.7, s=25, marker='s', label='Prediction')
                
                ax.set_title(f"Model Validation - {csv_key}", fontweight='bold')
                
                # [수정됨] Y축 레이블 변경 - 요청 반영 (Specific Capacity로 복구됨)
                ax.set_ylabel("Specific Capacity (mAh/g)")
                ax.set_xlabel("Cycle Number")
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
                if not pred.empty:
                    st.info(f"📊 **AI Report**: 최종 용량 **{pred['Capacity'].iloc[-1]:.2f} mAh/g** 예측됨.")
            else:
                st.warning(f"⚠️ 선택하신 '{csv_key}'에 대한 데이터를 찾을 수 없습니다.")
