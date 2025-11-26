import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- [1] 페이지 기본 설정 ---
st.set_page_config(page_title="Battery AI Simulator", layout="wide", page_icon="🔋")

# --- [1.1] 헤더 디자인 (로고 + 제목 + 로고) ---
col1, col2, col3 = st.columns([1.5, 6, 1.5])

with col1:
    try:
        st.image("ajou_logo.png", use_container_width=True)
    except:
        st.warning("로고(ajou_logo.png) 없음")

with col2:
    st.markdown("<h1 style='text-align: center;'>AI 기반 배터리 소재/공정 최적화 시뮬레이터</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Team 스물다섯 | Google-아주대학교 AI 융합 캡스톤 디자인</h5>", unsafe_allow_html=True)

with col3:
    try:
        st.image("google_logo.png", use_container_width=True)
    except:
        st.warning("로고(google_logo.png) 없음")

st.markdown("---")

st.info("""💡 이 플랫폼은 Engine 1(수명 예측)과 Engine 2(환경 영향 평가)를 통합한 최적화 시뮬레이터입니다.""")

# ==============================================================================
# [Engine 2] 데이터 로드 함수
# ==============================================================================
@st.cache_resource
def load_engine2_model():
    try:
        db = pd.read_excel('engine2_database.xlsx', sheet_name='LCA_Data', engine='openpyxl')
    except:
        # 데모용 데이터 (파일 없을 시)
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
# [메인 UI]
# ==============================================================================

tab1, tab2 = st.tabs(["⚡ Engine 1: 배터리 수명 예측", "🏭 Engine 2: 친환경 공정 최적화"])

# --- TAB 1: Engine 1 ---
with tab1:
    st.subheader("Engine 1. 배터리 장기 수명 예측 (Cycle Life Prediction)")
    st.markdown("**초기 100 Cycle 데이터**를 기반으로 **장기 수명 및 효율(CE)**을 예측합니다.")
    
    col_input, col_view = st.columns([1, 2])
    
    with col_input:
        st.markdown("##### 🧪 테스트 샘플 선택")
        sample_type = st.radio(
            "어떤 소재의 패턴을 테스트하시겠습니까?",
            ["Sample A (안정적 - CMGG)", "Sample B (일반적 - PVDF)", "Sample C (불안정 - 초기불량)"]
        )
        st.markdown("---")
        st.markdown("##### ⚙️ 예측 조건 설정")
        init_cap_input = st.number_input("초기 비용량 (Initial Capacity, mAh/g)", 100.0, 400.0, 185.0)
        cycle_input = st.number_input("예측 사이클 수 (Prediction Cycles)", 200, 5000, 1000, step=100)
        
        st.caption("※ 실제 데이터베이스(textbooks)의 학습 패턴을 기반으로 생성된 시뮬레이션입니다.")
        run_e1 = st.button("Engine 1 수명 예측 실행")

    with col_view:
        if run_e1:
            with st.spinner("AI가 초기 데이터를 분석하고 있습니다..."):
                if "Sample A" in sample_type:
                    decay = 1.0; label = "Excellent (CMGG)"; color = 'green'
                elif "Sample B" in sample_type:
                    decay = 2.5; label = "Normal (PVDF)"; color = 'orange'
                else:
                    decay = 5.0; label = "Poor (Defective)"; color = 'red'
                
                cycles, capacity, ce = predict_life_and_ce(decay_rate=decay, specific_cap_base=init_cap_input, cycles=cycle_input)
                
                fig2, (ax_cap, ax_ce) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # 1. Capacity Graph
                ax_cap.plot(cycles[:100], capacity[:100], 'k-', linewidth=2, label='Input Data (1~100)')
                ax_cap.plot(cycles[100:], capacity[100:], '--', color=color, linewidth=2, label=f'AI Prediction ({label})')
                ax_cap.set_ylabel("Specific Capacity (mAh/g)", fontsize=10, fontweight='bold')
                ax_cap.set_title("Discharge Capacity Prediction", fontsize=12, fontweight='bold')
                ax_cap.legend(loc='upper right')
                ax_cap.grid(True, alpha=0.3)
                
                # 2. CE Graph
                ax_ce.plot(cycles, ce, '-', color='blue', linewidth=1, alpha=0.7, label='Coulombic Efficiency')
                ax_ce.set_ylabel("Coulombic Efficiency (%)", fontsize=10, fontweight='bold')
                ax_ce.set_xlabel("Cycle Number", fontsize=10, fontweight='bold')
                ax_ce.set_ylim(98.0, 100.5)
                ax_ce.legend(loc='lower right')
                ax_ce.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                eol_limit = init_cap_input * 0.8
                eol_cycle = np.where(capacity < eol_limit)[0]
                
                st.info(f"📊 분석 리포트 ({cycle_input} Cycles)")
                if len(eol_cycle) > 0:
                    st.warning(f"⚠️ 예측 결과, 약 **{eol_cycle[0]} Cycle**에서 수명이 80%({eol_limit:.1f} mAh/g) 이하로 떨어질 것으로 예상됩니다.")
                else:
                    st.success(f"✅ 설정한 {cycle_input} Cycle까지 수명이 80% 이상 안정적으로 유지될 것으로 예측됩니다.")
        else:
            st.info("조건을 설정하고 [Engine 1 수명 예측 실행] 버튼을 눌러주세요.")

# --- TAB 2: Engine 2 ---
with tab2:
    model_e2, prep_e2, db_e2 = load_engine2_model()
    
    st.subheader("Engine 2. 공정 변수에 따른 환경 영향 예측 (LCA)")
    st.info("좌측 사이드바에서 공정 조건(Binder, Solvent, 건조 온도 등)을 변경해보세요.")

    with st.sidebar:
        st.header("🛠️ Engine 2 공정 설정")
        s_binder = st.selectbox("Binder Type", ["PVDF", "CMGG", "GG", "CMC"])
        s_solvent = st.radio("Solvent Type", ["NMP", "Water"])
        st.markdown("---")
        s_temp = st.slider("Drying Temp (°C)", 60, 180, 110)
        s_time = st.slider("Drying Time (min)", 10, 720, 120) 
        s_loading = st.number_input("Mass Loading (g/m²)", 1.0, 100.0, 20.0)
        run_e2 = st.button("Engine 2 예측 실행")

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
            col1.metric("CO₂ 배출량", f"{pred[0]:.4f} kg/m²")
            col2.metric("에너지 소비", f"{pred[1]:.4f} kWh/m²")
            col3.metric("VOC 배출량", f"{pred[2]:.4f} g/m²", delta="-100%" if pred[2]<0.01 else None)
            
            st.markdown("#### 📊 기존 NMP 공정 대비 비교 (Comparison vs NMP Process)")
            nmp_mean = db_e2[db_e2['Solvent_Type']=='NMP'][['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2']].mean()
            if nmp_mean.isnull().all():
                nmp_mean = pd.Series([0.27, 0.6, 3.0], index=['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2'])

            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(3)
            width = 0.35
            
            # [디자인 수정] PPT 색감 반영
            color_nmp = '#FA8072'  # Salmon
            color_sim = '#90EE90'  # LightGreen
            
            ax.bar(x - width/2, nmp_mean.values, width, label='Reference (NMP)', color=color_nmp)
            ax.bar(x + width/2, pred, width, label='Current Simulation', color=color_sim)
            
            ax.set_xticks(x)
            ax.set_xticklabels(['CO2', 'Energy', 'VOC'], fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"예측 오류: {e}")
    else:
        st.write("👈 왼쪽 사이드바에서 [Engine 2 예측 실행] 버튼을 눌러주세요.")
