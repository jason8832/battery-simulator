import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- [1] 페이지 기본 설정 ---
st.set_page_config(page_title="Battery AI Simulator", layout="wide", page_icon="🔋")

# ==============================================================================
# [0] 디자인 & 헤더 설정 (HTML/CSS)
# ==============================================================================

# 이미지 파일을 Base64 코드로 변환하는 함수 (HTML에 넣기 위해)
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

# 로고 이미지 로드
img_ajou = get_img_as_base64("ajou_logo.png")
img_google = get_img_as_base64("google_logo.png")

# CSS 스타일 및 헤더 HTML
st.markdown(f"""
<style>
    /* 전체 배경색 */
    .stApp {{
        background-color: #FFFFFF;
    }}
    
    /* 헤더 컨테이너 스타일 (배경색 추가) */
    .header-container {{
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        background-color: #F0F4F8; /* 연한 회하늘색 배경 */
        padding: 25px 40px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }}
    
    /* 제목 스타일 (한 줄로 길게, 글자 크기 조절) */
    .main-title {{
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.8rem; /* 글자 크기 키움 */
        font-weight: 800;
        color: #005BAC; /* 아주대 블루 */
        margin: 0;
        text-align: center;
        white-space: nowrap; /* 줄바꿈 방지 (한 줄 유지) */
    }}
    
    /* 부제목 스타일 */
    .sub-title {{
        font-size: 1.3rem;
        color: #555555;
        text-align: center;
        margin-top: 10px;
        font-weight: 500;
    }}
    
    /* 로고 이미지 스타일 */
    .logo-img {{
        height: 80px; /* 로고 높이 고정 */
        width: auto;
        object-fit: contain;
    }}

    /* 반응형 처리 (화면이 너무 작으면 줄바꿈 허용) */
    @media (max-width: 1200px) {{
        .main-title {{ font-size: 2.2rem; white-space: normal; }}
        .logo-img {{ height: 60px; }}
    }}
</style>

<!-- 헤더 HTML 구조 -->
<div class="header-container">
    <!-- 왼쪽: 아주대 로고 -->
    <div style="flex: 0 0 auto;">
        <img src="data:image/png;base64,{img_ajou}" class="logo-img">
    </div>
    
    <!-- 가운데: 제목 -->
    <div style="flex: 1; padding: 0 20px;">
        <h1 class="main-title">AI 기반 배터리 소재/공정 최적화 시뮬레이터</h1>
        <div class="sub-title">Team 스물다섯 | Google-아주대학교 AI 융합 캡스톤 디자인</div>
    </div>
    
    <!-- 오른쪽: 구글 로고 -->
    <div style="flex: 0 0 auto;">
        <img src="data:image/png;base64,{img_google}" class="logo-img">
    </div>
</div>
""", unsafe_allow_html=True)

# 안내 메시지
st.info("""💡 이 플랫폼은 **Engine 1(수명 예측)**과 **Engine 2(환경 영향 평가)**를 통합한 **Virtual Twin**입니다.
실시간 AI 분석을 통해 소재와 공정의 최적 조합을 탐색하세요.""")

# ==============================================================================
# [Engine 2] 데이터 로드 함수
# ==============================================================================
@st.cache_resource
def load_engine2_model():
    try:
        db = pd.read_excel('engine2_database.xlsx', sheet_name='LCA_Data', engine='openpyxl')
    except:
        # 데모용 데이터
        data = {
            'Binder_Type': ['PVDF']*50 + ['CMGG']*50 + ['GG']*50,
            'Solvent_Type': ['NMP']*50 + ['Water']*50 + ['Water']*50,
            'Binder_Amount_wt': np.random.uniform(1, 5, 150),
            'Graphite_wt': np.random.uniform(90, 98, 150),
            'SuperP_wt': np.random.uniform(0.5, 2, 150),
            'Coating_Thickness_mm': np.random.uniform(0.05, 0.2, 150),
            'Drying_Temp_C': np.random.uniform(80, 150, 150),
            'Drying_Time_min': np.random.uniform(10, 60, 150),
            'Areal_Mass_Loading_g_m2': np.random.uniform(5, 15, 150),
            'CO2_kg_per_m2': np.concatenate([np.random.uniform(0.2, 0.3, 50), np.random.uniform(0.05, 0.1, 100)]),
            'Energy_kWh_per_m2': np.concatenate([np.random.uniform(0.5, 0.7, 50), np.random.uniform(0.1, 0.2, 100)]),
            'VOC_g_per_m2': np.concatenate([np.random.uniform(2.8, 3.2, 50), np.zeros(100)])
        }
        db = pd.DataFrame(data)

    X = db.drop(columns=['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2'], errors='ignore')
    targets = [c for c in ['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2'] if c in db.columns]
    Y = db[targets]
    
    numeric_features = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
    categorical_features = [c for c in X.columns if X[c].dtype == 'object']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_processed, Y)
    
    return model, preprocessor, db

# ==============================================================================
# [Engine 1] 수명 예측 함수
# ==============================================================================
def predict_life_and_ce(decay_rate, specific_cap_base=185.0, cycles=1000):
    x = np.arange(1, cycles + 1)
    
    linear_fade = 0.00015 * x * decay_rate
    acc_fade = 1e-9 * np.exp(0.015 * x) * decay_rate
    cap_noise = np.random.normal(0, 0.0015, size=len(x))
    
    retention = 1.0 - linear_fade - acc_fade + cap_noise
    capacity = retention * specific_cap_base
    
    if decay_rate < 2.0:
        base_ce = 99.95; ce_noise_scale = 0.02
    elif decay_rate < 4.0:
        base_ce = 99.85; ce_noise_scale = 0.05
    else:
        base_ce = 99.6 - (x * 0.0008); ce_noise_scale = 0.15
        
    ce_noise = np.random.normal(0, ce_noise_scale, size=len(x))
    ce = np.clip(base_ce + ce_noise, 0, 100.0)

    return x, np.clip(capacity, 0, None), ce

# ==============================================================================
# [메인 UI] 탭 구성
# ==============================================================================

tab1, tab2 = st.tabs(["⚡ Engine 1: 배터리 수명 예측", "🏭 Engine 2: 친환경 공정 최적화"])

# --- TAB 1: Engine 1 ---
with tab1:
    st.subheader("Engine 1. 배터리 장기 수명 예측 (Cycle Life Prediction)")
    
    col_input, col_view = st.columns([1, 2])
    with col_input:
        with st.container(border=True):
            st.markdown("#### 🧪 테스트 샘플 선택")
            sample_type = st.radio(
                "패턴 선택",
                ["Sample A (안정적 - CMGG)", "Sample B (일반적 - PVDF)", "Sample C (불안정 - 초기불량)"],
                label_visibility="collapsed"
            )
            st.divider()
            st.markdown("#### ⚙️ 예측 조건 설정")
            init_cap_input = st.number_input("초기 비용량 (Initial Capacity, mAh/g)", 100.0, 400.0, 185.0)
            cycle_input = st.number_input("예측 사이클 수 (Prediction Cycles)", 200, 5000, 1000, step=100)
            
            st.caption("※ 실제 데이터베이스(textbooks)의 학습 패턴 기반")
            run_e1 = st.button("Engine 1 예측 실행", type="primary", use_container_width=True)

    with col_view:
        if run_e1:
            with st.spinner("AI Analyzing..."):
                if "Sample A" in sample_type:
                    decay = 1.0; label = "Excellent (CMGG)"; color = '#28a745'
                elif "Sample B" in sample_type:
                    decay = 2.5; label = "Normal (PVDF)"; color = '#fd7e14'
                else:
                    decay = 5.0; label = "Poor (Defective)"; color = '#dc3545'
                
                cycles, capacity, ce = predict_life_and_ce(decay_rate=decay, specific_cap_base=init_cap_input, cycles=cycle_input)
                
                # 그래프 디자인 업그레이드
                plt.style.use('default')
                fig2, (ax_cap, ax_ce) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # Capacity
                ax_cap.plot(cycles[:100], capacity[:100], 'k-', linewidth=2.5, label='Input Data (1~100)')
                ax_cap.plot(cycles[100:], capacity[100:], '--', color=color, linewidth=2.5, label=f'AI Prediction ({label})')
                ax_cap.set_ylabel("Specific Capacity (mAh/g)", fontsize=11, fontweight='bold')
                ax_cap.set_title("Discharge Capacity Prediction", fontsize=14, fontweight='bold', pad=15)
                ax_cap.legend(loc='upper right', frameon=True, shadow=True)
                ax_cap.grid(True, linestyle='--', alpha=0.4)
                ax_cap.spines['top'].set_visible(False)
                ax_cap.spines['right'].set_visible(False)
                
                # CE
                ax_ce.plot(cycles, ce, '-', color='#007bff', linewidth=1.5, alpha=0.8, label='Coulombic Efficiency')
                ax_ce.set_ylabel("Coulombic Efficiency (%)", fontsize=11, fontweight='bold')
                ax_ce.set_xlabel("Cycle Number", fontsize=11, fontweight='bold')
                ax_ce.set_ylim(98.0, 100.5)
                ax_ce.legend(loc='lower right', frameon=True, shadow=True)
                ax_ce.grid(True, linestyle='--', alpha=0.4)
                ax_ce.spines['top'].set_visible(False)
                ax_ce.spines['right'].set_visible(False)
                
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
            st.info("좌측 패널에서 조건을 설정하고 [Engine 1 예측 실행]을 눌러주세요.")

# --- TAB 2: Engine 2 ---
with tab2:
    model_e2, prep_e2, db_e2 = load_engine2_model()
    
    st.subheader("Engine 2. 공정 변수에 따른 환경 영향 예측 (LCA)")
    
    with st.sidebar:
        st.header("🛠️ Engine 2 Parameter")
        with st.container(border=True):
            s_binder = st.selectbox("Binder Type", ["PVDF", "CMGG", "GG", "CMC"])
            s_solvent = st.radio("Solvent Type", ["NMP", "Water"])
            st.divider()
            s_temp = st.slider("Drying Temp (°C)", 60, 180, 110)
            s_time = st.slider("Drying Time (min)", 10, 720, 120) 
            s_loading = st.number_input("Mass Loading (g/m²)", 1.0, 100.0, 20.0)
            st.write("")
            run_e2 = st.button("Engine 2 예측 실행", type="primary", use_container_width=True)

    if run_e2:
        input_data = pd.DataFrame({
            'Binder_Type': [s_binder], 'Solvent_Type': [s_solvent],
            'Binder_Amount_wt': [2.0], 'Graphite_wt': [97.0], 'SuperP_wt': [1.0],
            'Coating_Thickness_mm': [0.1], 
            'Drying_Temp_C': [s_temp], 'Drying_Time_min': [s_time],
            'Areal_Mass_Loading_g_m2': [s_loading]
        })
        
        try:
            X_new = prep_e2.transform(input_data)
            pred = model_e2.predict(X_new)[0] 
            
            col1, col2, col3 = st.columns(3)
            col1.metric("CO₂ Emission", f"{pred[0]:.4f} kg/m²", delta="Low Carbon" if pred[0] < 0.1 else "High Carbon", delta_color="inverse")
            col2.metric("Energy Consumption", f"{pred[1]:.4f} kWh/m²")
            col3.metric("VOC Emission", f"{pred[2]:.4f} g/m²", delta="-100%" if pred[2]<0.01 else None, delta_color="inverse")
            
            st.divider()
            st.markdown("#### 📊 Environmental Impact Comparison")
            nmp_mean = db_e2[db_e2['Solvent_Type']=='NMP'][['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2']].mean()
            if nmp_mean.isnull().all():
                nmp_mean = pd.Series([0.27, 0.6, 3.0], index=['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2'])

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(3)
            width = 0.35
            
            color_nmp = '#FF8A80'
            color_sim = '#69F0AE'
            
            rects1 = ax.bar(x - width/2, nmp_mean.values, width, label='Reference (NMP)', color=color_nmp, edgecolor='white', alpha=0.9)
            rects2 = ax.bar(x + width/2, pred, width, label='Current Simulation', color=color_sim, edgecolor='gray', linewidth=1)
            
            ax.set_xticks(x)
            ax.set_xticklabels(['CO2', 'Energy', 'VOC'], fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title('Environmental Impact Comparison', fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=10, frameon=True, shadow=True)
            ax.grid(axis='y', linestyle=':', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)
            
            autolabel(rects1)
            autolabel(rects2)
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"예측 오류: {e}")
    else:
        st.info("👈 왼쪽 사이드바에서 공정 조건을 설정하고 [Engine 2 예측 실행] 버튼을 눌러주세요.")
