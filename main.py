import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("생태계 개체수 변화 시뮬레이터")

st.write("""
이 시뮬레이터는 생산자, 1차 소비자, 2차 소비자, 분해자의 개체수 변화를 보여줍니다.
각 개체군의 초기 개체수를 조절하여 환경 변화를 예측해보세요!
""")

# --- 1. 사이드바에 입력 위젯 배치 ---
st.sidebar.header("초기 개체수 설정")

initial_producer = st.sidebar.slider("생산자 초기 개체수", 100, 1000, 500, 50)
initial_primary_consumer = st.sidebar.slider("1차 소비자 초기 개체수", 10, 500, 100, 20)
initial_secondary_consumer = st.sidebar.slider("2차 소비자 초기 개체수", 1, 200, 30, 10)
initial_decomposer = st.sidebar.slider("분해자 초기 개체수", 50, 500, 200, 25)

simulation_steps = st.sidebar.slider("시뮬레이션 시간 (단계)", 50, 500, 200, 10)

# --- 2. 생태계 모델 함수 (예시: 간단한 로직) ---
def simulate_ecosystem(p_init, pc_init, sc_init, d_init, steps):
    producers = [p_init]
    primary_consumers = [pc_init]
    secondary_consumers = [sc_init]
    decomposers = [d_init]

    # 각 개체군 변화율 (이 값들을 조절하여 다양한 시나리오 구현 가능)
    producer_growth = 0.05
    producer_decay_by_pc = 0.0005 # 1차 소비자가 생산자를 먹는 비율

    pc_growth_by_p = 0.0002 # 1차 소비자가 생산자를 먹고 성장하는 비율
    pc_decay = 0.02
    pc_decay_by_sc = 0.001 # 2차 소비자가 1차 소비자를 먹는 비율

    sc_growth_by_pc = 0.0001 # 2차 소비자가 1차 소비자를 먹고 성장하는 비율
    sc_decay = 0.03

    decomposer_growth_from_decay = 0.005 # 죽은 개체로부터 분해자 성장

    for i in range(1, steps):
        # 이전 단계 개체수
        p = producers[-1]
        pc = primary_consumers[-1]
        sc = secondary_consumers[-1]
        d = decomposers[-1]

        # 생산자 변화
        next_p = p * (1 + producer_growth) - (p * pc * producer_decay_by_pc)
        next_p = max(0, next_p) # 개체수가 음수가 되지 않도록
        
        # 1차 소비자 변화
        next_pc = pc * (1 - pc_decay) + (p * pc * pc_growth_by_p) - (pc * sc * pc_decay_by_sc)
        next_pc = max(0, next_pc)

        # 2차 소비자 변화
        next_sc = sc * (1 - sc_decay) + (pc * sc * sc_growth_by_pc)
        next_sc = max(0, next_sc)
        
        # 분해자 변화 (죽은 개체수에 비례하여 증가, 자체 소멸율도 고려 가능)
        # 단순화를 위해 각 개체군 감소분에 비례하여 분해자가 늘어난다고 가정
        decayed_matter = (p - next_p) + (pc - next_pc) + (sc - next_sc) # 죽은 개체량 추정
        next_d = d + (decayed_matter * decomposer_growth_from_decay)
        next_d = max(50, next_d) # 분해자는 일정 수준 이하로 떨어지지 않도록 (혹은 최소값 설정)

        producers.append(next_p)
        primary_consumers.append(next_pc)
        secondary_consumers.append(next_sc)
        decomposers.append(next_d)

    return pd.DataFrame({
        '시간': range(steps),
        '생산자': producers,
        '1차 소비자': primary_consumers,
        '2차 소비자': secondary_consumers,
        '분해자': decomposers
    })

# --- 3. 시뮬레이션 실행 및 그래프 그리기 ---
if st.sidebar.button("시뮬레이션 시작"):
    st.subheader("개체수 변화 그래프")
    
    # 시뮬레이션 실행
    df_results = simulate_ecosystem(
        initial_producer,
        initial_primary_consumer,
        initial_secondary_consumer,
        initial_decomposer,
        simulation_steps
    )
    
    # Streamlit 기본 라인 차트
    # st.line_chart(df_results.set_index('시간'))

    # Matplotlib을 이용한 그래프 (더 많은 커스터마이징 가능)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_results['시간'], df_results['생산자'], label='생산자', color='green')
    ax.plot(df_results['시간'], df_results['1차 소비자'], label='1차 소비자', color='blue')
    ax.plot(df_results['시간'], df_results['2차 소비자'], label='2차 소비자', color='red')
    ax.plot(df_results['시간'], df_results['분해자'], label='분해자', color='brown')
    
    ax.set_xlabel("시간 단계")
    ax.set_ylabel("개체수")
    ax.set_title("시간에 따른 개체수 변화")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig) # Streamlit에 Matplotlib 그래프 표시

    st.subheader("시뮬레이션 결과 데이터")
    st.dataframe(df_results)
else:
    st.info("왼쪽 사이드바에서 초기 개체수를 설정하고 '시뮬레이션 시작' 버튼을 눌러주세요.")
