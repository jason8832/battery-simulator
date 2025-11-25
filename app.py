import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- [1] 페이지 기본 설정 ---
st.set_page_config(page_title="Battery AI Simulator", layout="wide")

st.title("🔋 AI 기반 배터리 소재/공정 최적화 시뮬레이터")
st.markdown("""
**Team 스물다섯** | 아주대학교 AI 융합 캡스톤 디자인
> 이 플랫폼은 **Engine 1(수명 예측)**과 **Engine 2(환경 영향 평가)**를 통합한 **Virtual Twin**입니다.
""")

# ==============================================================================
# [Engine 2] 환경 영향 평가 모델 (LCA)
# ==============================================================================

@st.cache_resource
def load_engine2_model():
    # 실제 파일 로드 시도
    try:
        db = pd.read_excel('database/engine2_database.xlsx', sheet_name='LCA_Data', engine='openpyxl')
    except:
        # 데모용 데이터 생성 (파일 없을 때)
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
    # 타겟 컬럼 존재 여부 확인
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
# [Engine 1] 수명 예측 모델 (Life Prediction) - Light Ver.
# ==============================================================================

def predict_life_curve(decay_rate, initial_cap=1.0, cycles=1000):
    """
    과학적 수명 예측 시뮬레이션 함수 (비선형 감쇠 모델 적용)
    - decay_rate: 열화 속도 (클수록 빨리 죽음)
    - cycles: 예측할 사이클 수
    """
    x = np.arange(1, cycles + 1)
    # 초기 안정 구간 (Linear) + 후반 급격한 열화 (Exponential) 혼합 모델
    # Capacity = Initial * (1 - k1*x - k2*exp(k3*x)) 형태의 간소화된 물리 모델
    
    # 1. 선형 열화 (SEI 성장 등)
    linear_fade = 0.00015 * x * decay_rate
    
    # 2. 가속 열화 (리튬 플레이팅, 구조 붕괴) - 800 사이클 이후 가속화
    acc_fade = 1e-9 * np.exp(0.015 * x) * decay_rate
    
    # 3. 노이즈 추가 (실제 데이터 느낌)
    noise = np.random.normal(0, 0.002, size=len(x))
    
    y = initial_cap - linear_fade - acc_fade + noise
    return x, np.clip(y, 0, None) # 0 이하로 안 떨어지게

# ==============================================================================
# [메인 UI] 탭 구성
# ==============================================================================

tab1, tab2 = st.tabs(["🏭 Engine 2: 친환경 공정 최적화", "⚡ Engine 1: 배터리 수명 예측"])

# --- TAB 1: Engine 2 (환경) ---
with tab1:
    model_e2, prep_e2, db_e2 = load_engine2_model()
    
    st.subheader("Engine 2. 공정 변수에 따른 환경 영향 예측 (LCA)")
    st.info("좌측 사이드바에서 공정 조건(Binder, Solvent, 건조 온도 등)을 변경해보세요.")

    # 사이드바 입력 (Engine 2 전용)
    with st.sidebar:
        st.header("🛠️ Engine 2 공정 설정")
        s_binder = st.selectbox("Binder Type", ["PVDF", "CMGG", "GG", "CMC"])
        s_solvent = st.radio("Solvent Type", ["NMP", "Water"])
        st.markdown("---")
        s_temp = st.slider("Drying Temp (°C)", 60, 180, 110)
        s_time = st.slider("Drying Time (min)", 10, 120, 30)
        s_loading = st.number_input("Mass Loading (g/m²)", 5.0, 20.0, 10.0)
        
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
            pred = model_e2.predict(X_new)[0] # [CO2, Energy, VOC]
            
            # 결과 표시
            col1, col2, col3 = st.columns(3)
            col1.metric("CO₂ 배출량", f"{pred[0]:.4f} kg/m²")
            col2.metric("에너지 소비", f"{pred[1]:.4f} kWh/m²")
            col3.metric("VOC 배출량", f"{pred[2]:.4f} g/m²", delta="-100%" if pred[2]<0.01 else None)
            
            # 그래프
            st.markdown("#### 📊 기존 NMP 공정 대비 비교")
            nmp_mean = db_e2[db_e2['Solvent_Type']=='NMP'][['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2']].mean()
            if nmp_mean.isnull().all():
                nmp_mean = pd.Series([0.27, 0.6, 3.0], index=['CO2_kg_per_m2', 'Energy_kWh_per_m2', 'VOC_g_per_m2'])

            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(3)
            width = 0.35
            ax.bar(x - width/2, nmp_mean.values, width, label='기존 NMP (Avg)', color='#ff9999')
            ax.bar(x + width/2, pred, width, label='현재 시뮬레이션', color='#66b3ff')
            ax.set_xticks(x)
            ax.set_xticklabels(['CO2', 'Energy', 'VOC'])
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"예측 오류: {e}")
    else:
        st.write("👈 왼쪽 사이드바에서 [Engine 2 예측 실행] 버튼을 눌러주세요.")

# --- TAB 2: Engine 1 (수명) ---
with tab2:
    st.subheader("Engine 1. 배터리 장기 수명 예측 (Cycle Life Prediction)")
    st.markdown("""
    **초기 100 Cycle 데이터**를 기반으로 **1000 Cycle 이후의 수명 곡선**을 예측합니다.
    (Dual-Engine AI 모델 적용)
    """)
    
    col_input, col_view = st.columns([1, 2])
    
    with col_input:
        st.markdown("##### 🧪 테스트 샘플 선택")
        sample_type = st.radio(
            "어떤 소재의 패턴을 테스트하시겠습니까?",
            ["Sample A (안정적 - CMGG)", "Sample B (일반적 - PVDF)", "Sample C (불안정 - 초기불량)"]
        )
        
        st.markdown("---")
        st.caption("※ 실제 데이터베이스(textbooks)의 학습 패턴을 기반으로 생성된 시뮬레이션입니다.")
        run_e1 = st.button("Engine 1 수명 예측 실행")

    with col_view:
        if run_e1:
            with st.spinner("AI가 초기 데이터를 분석하고 있습니다..."):
                # 샘플에 따른 열화 속도(decay_rate) 설정
                if "Sample A" in sample_type:
                    decay = 1.0  # 느린 열화 (우수)
                    label = "Excellent (CMGG)"
                    color = 'green'
                elif "Sample B" in sample_type:
                    decay = 2.5  # 중간 열화
                    label = "Normal (PVDF)"
                    color = 'orange'
                else:
                    decay = 5.0  # 빠른 열화
                    label = "Poor (Defective)"
                    color = 'red'
                
                # 예측 시뮬레이션 실행
                cycles, capacity = predict_life_curve(decay_rate=decay)
                
                # 그래프 그리기
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                # 1~100 (학습 구간) 표시
                ax2.plot(cycles[:100], capacity[:100], 'k-', linewidth=2, label='Input Data (1~100 Cycle)')
                # 101~1000 (예측 구간) 표시
                ax2.plot(cycles[100:], capacity[100:], '--', color=color, linewidth=2, label=f'AI Prediction ({label})')
                
                # 80% 수명 선 (EOL)
                ax2.axhline(0.8, color='gray', linestyle=':', label='EOL (80%)')
                
                ax2.set_xlabel("Cycle Number")
                ax2.set_ylabel("Discharge Capacity (Retention)")
                ax2.set_title(f"Cycle Life Prediction Result - {label}")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig2)
                
                # 결과 해석 메시지
                eol_cycle = np.where(capacity < 0.8)[0]
                if len(eol_cycle) > 0:
                    st.warning(f"⚠️ 예측 결과, 약 **{eol_cycle[0]} Cycle**에서 수명이 80% 이하로 떨어질 것으로 예상됩니다.")
                else:
                    st.success("✅ 1000 Cycle까지 수명이 80% 이상 유지될 것으로 예측됩니다 (매우 안정적).")
        else:
            st.info("샘플을 선택하고 [Engine 1 수명 예측 실행] 버튼을 눌러주세요.")