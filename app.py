import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import folium
from streamlit_folium import st_folium
import numpy as np

# 1. 기본 설정 및 테마
st.set_page_config(page_title="서울 카페 리스크 인텔리전스", layout="wide", initial_sidebar_state="expanded")

# 폰트 설정 (Windows, Mac, Linux/Streamlit Cloud 공용)
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin': # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    # 스트림릿 클라우드(리눅스) 환경
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False

# 커스텀 CSS (글자색 강제 지정 및 카드 디자인)
st.markdown("""
    <style>
    /* 메인 배경색 */
    .main { background-color: #f0f2f6; }
    
    /* Metric 박스 스타일 */
    [data-testid="stMetricValue"] {
        color: #1f77b4 !important; /* 숫자 색상 (파란색 계열) */
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #333333 !important; /* 라벨 색상 (진한 회색) */
        font-weight: bold !important;
    }
    
    /* Metric 박스 테두리 및 배경 */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }

    /* 탭 메뉴 글자색 */
    .stTabs [data-baseweb="tab"] p {
        color: #444444;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 데이터 로드 (불필요한 pickle 및 merge 제거)
@st.cache_data
def load_data():
    # 전처리 및 지표가 통합된 최종 파일을 읽어옵니다.
    # 파일명이 app_data.csv 인지 꼭 확인하세요!
    df = pd.read_csv("model/app_data.csv", encoding='utf-8-sig')
    
    # 행정동명 공백 제거 (안전 장치)
    df['행정동명'] = df['행정동명'].str.strip()
    
    # 혹시라도 위경도가 없는 행이 있다면 제거하거나 기본값 부여
    df = df.dropna(subset=['위도', '경도'])
    
    return df

df_final = load_data()

# ---------------------
# Sidebar
# ---------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/coffee-to-go.png", width=80)
    st.title("Cafe Risk AI")
    st.markdown("---")
    menu = st.radio("📑 분석 메뉴", ["전체 대시보드", "리스크 지도", "상세 지역 진단"], index=0)
    
    st.markdown("### 🛠️ 필터 설정")
    risk_filter = st.multiselect("위험 등급", ["위험", "주의", "안전"], default=["위험", "주의", '안전'])
    # 1. 슬라이더 범위를 실제 데이터의 최대값에 맞춤
    max_val = float(df_final['예측_폐업률'].max())
    min_rate = st.sidebar.slider("최소 폐업률 (%)", 0.0, max_val, 0.0) # 20 대신 max_val 사용
    
    st.markdown("---")
    st.caption(f"최종 업데이트: 2026-04-02\nAnalyst: Han Sujeong")

filtered_df = df_final[
    (df_final['위험등급'].isin(risk_filter)) & 
    (df_final['예측_폐업률'] >= min_rate)
]

# ---------------------
# PAGE 1: 전체 상권 리포트
# ---------------------
if menu == "전체 대시보드":
    st.title("🏙️ 서울시 카페 폐업 리스크 현황")
    
    if filtered_df.empty:
        st.warning("⚠️ 선택하신 조건에 맞는 데이터가 없습니다. 필터를 조절해 주세요.")
    else:
        # 1. Metric 카드
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("분석 대상", f"{len(filtered_df)}개 동", "Seoul")
        with m2:
            avg_v = filtered_df['예측_폐업률'].mean()
            st.metric("평균 예측 폐업률", f"{avg_v:.2f}%", f"{avg_v-5:.1f}%", delta_color="inverse")
        with m3:
            high_d = filtered_df.sort_values('예측_폐업률', ascending=False).iloc[0]['행정동명']
            st.metric("최고 위험 지역", high_d, "Warning")
        with m4:
            st.metric("모델 신뢰도", "R² 0.74", "Stable")

        st.markdown("---")

        # 2. 메인 분석 영역
        tab1, tab2 = st.tabs(["🔥 위험 지역 랭킹", "📊 자치구별 통계"])
        
        with tab1:
            st.subheader("폐업 위험도 TOP 10")
            # 데이터프레임 시각화 강화
            st.dataframe(
                filtered_df.sort_values('예측_폐업률', ascending=False).head(10)[['시군구명', '행정동명', '예측_폐업률', '위험등급', '리스크_이유']],
                column_config={
                    "예측_폐업률": st.column_config.ProgressColumn("예측 폐업률", format="%.2f%%", min_value=0, max_value=20),
                    "위험등급": st.column_config.TextColumn("상태", width="small"),
                    "리스크_이유": st.column_config.TextColumn("주요 원인", width="large")
                },
                use_container_width=True,
                hide_index=True
            )

        with tab2:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.subheader("자치구별 평균 폐업률")
                gu_avg = filtered_df.groupby('시군구명')['예측_폐업률'].mean().sort_values()
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = sns.color_palette("Reds", len(gu_avg))
                gu_avg.plot(kind='barh', color=colors, ax=ax)
                ax.set_title("자치구별 리스크 지수")
                st.pyplot(fig)
            with col_c2:
                st.subheader("인프라(집객시설) vs 폐업률")
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                sns.regplot(data=filtered_df, x='집객시설_수', y='예측_폐업률', scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax2)
                st.pyplot(fig2)

# ---------------------
# PAGE 2: 리스크 지도
# ---------------------
elif menu == "리스크 지도":
    st.title("🗺️ 리스크 지점 정밀 탐색")
    
    c_map, c_legend = st.columns([4, 1])
    
    with c_legend:
        st.info("**지도 범례**")
        st.write("🔴 **위험**: 폐업률 상위 10%")
        st.write("🟠 **주의**: 폐업률 상위 30%")
        st.write("🟢 **안전**: 기타 지역")
        st.caption("원의 크기는 예측 폐업률에 비례합니다.")

    with c_map:
        m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles='cartodbpositron')
        for i, row in filtered_df.iterrows():
            if pd.isna(row['위도']): continue
            color = 'red' if row['위험등급'] == '위험' else 'orange' if row['위험등급'] == '주의' else 'green'
            popup_html = f"""<div style='width:180px'><b>{row['행정동명']}</b><br>예측: {row['예측_폐업률']:.2f}%<br><small>{row['리스크_이유']}</small></div>"""
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=row['예측_폐업률'] * 1.5,
                color=color, fill=True, fill_opacity=0.6,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)
        st_folium(m, width=1000, height=600)

# ---------------------
# PAGE 3: 상세 분석
# ---------------------
elif menu == "상세 지역 진단":
    st.title("🔍 지역별 심층 리포트")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        sel_gu = st.selectbox("자치구 선택", sorted(df_final['시군구명'].unique()))
    with col_sel2:
        sel_dong = st.selectbox("행정동 선택", sorted(df_final[df_final['시군구명']==sel_gu]['행정동명'].unique()))
    
    data = df_final[(df_final['시군구명']==sel_gu) & (df_final['행정동명']==sel_dong)].iloc[0]
    
    st.markdown("---")
    
    res_c1, res_c2, res_c3 = st.columns([1, 1, 2])
    res_c1.metric("최종 예측 폐업률", f"{data['예측_폐업률']:.3f}%")
    
    if data['위험등급'] == '위험':
        res_c2.error(f"등급: {data['위험등급']}")
    elif data['위험등급'] == '주의':
        res_c2.warning(f"등급: {data['위험등급']}")
    else:
        res_c2.success(f"등급: {data['위험등급']}")
        
    res_c3.info(f"**💡 리스크 원인 진단**\n\n{data['리스크_이유']}")
# ---------------------
    # PAGE 3: 상세 분석 (차트 부분 업데이트)
    # ---------------------
    st.subheader("📍 서울 평균 대비 지표 분석")
    
    # 데이터 준비
    metrics_map = {
        '카페밀도지수': '카페밀도',
        '유동인구수': '유동인구',
        '집객시설_수': '인프라',
        '유사업종비율': '경쟁업종'
    }
    
    # 수치 비교 데이터프레임 생성
    comp_data = []
    for eng, kor in metrics_map.items():
        comp_data.append({
            "지표": kor,
            "해당 지역": data[eng],
            "서울 평균": df_final[eng].mean()
        })
    df_comp = pd.DataFrame(comp_data)
# ---------------------
    # PAGE 3: 상세 분석 (정규화 차트 버전)
    # ---------------------
    # (이전 df_comp 만드는 로직은 그대로 둡니다)
    
    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        # 🌟 정규화 로직 추가: 유동인구가 너무 커서 차트가 안 보이므로 단위를 맞춥니다.
        # 서울 전체 데이터의 최솟값/최댓값을 가져와서 0~1 사이로 환산합니다.
        df_norm = df_comp.copy()
        for col_name in ['해당 지역', '서울 평균']:
            for metrics_col in ['카페밀도지수', '유동인구수', '집객시설_수', '유사업종비율']:
                min_v = df_final[metrics_col].min()
                max_v = df_final[metrics_col].max()
                
                # 0~1 정규화 (Min-Max Scaling)
                # df_comp의 값을 바로 쓰지 않고 data 행에서 다시 계산합니다.
                if col_name == '해당 지역':
                    val = data[metrics_col]
                else:
                    val = df_final[metrics_col].mean()
                
                # 병합용으로 kor name을 사용
                norm_v = (val - min_v) / (max_v - min_v + 1e-6) # 0으로 나누기 방지
                
                # kor name 매핑
                eng_to_kor = {
                    '카페밀도지수': '카페밀도',
                    '유동인구수': '유동인구',
                    '집객시설_수': '인프라',
                    '유사업종비율': '경쟁업종'
                }
                
                idx = df_norm[df_norm['지표'] == eng_to_kor[metrics_col]].index[0]
                df_norm.loc[idx, col_name] = norm_v # 값만 교체

        # matplotlib으로 정규화 차트 그리기
        fig_norm, ax_norm = plt.subplots(figsize=(10, 6))
        df_melted_norm = df_norm.melt(id_vars="지표", var_name="구분", value_name="상대 점수")
        
        # 그래프 스타일 통일 (다크모드에서도 잘 보이도록)
        sns.barplot(data=df_melted_norm, x="지표", y="상대 점수", hue="구분", palette="coolwarm", ax=ax_norm)
        
        # 막대 위에 값 표시 (0~1 사이 점수)
        for p in ax_norm.patches:
            ax_norm.annotate(f'{p.get_height():.2f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points',
                            fontsize=9)
        
        ax_norm.set_title(f"{sel_dong} 상권 지표 분석 (상대 평가)")
        ax_norm.set_ylabel("상대 점수 (0 ~ 1)")
        ax_norm.set_ylim(0, 1.1) # Y축 범위 고정
        st.pyplot(fig_norm)

    with table_col:
        st.markdown("#### 📋 상세 수치 (실제 값)")
        st.dataframe(df_comp.set_index("지표"), use_container_width=True)
        st.info("💡 위 차트는 서울 전체 대비 상대 점수입니다. 실제 수치는 표를 확인하세요.")

st.success("대시보드가 성공적으로 로드되었습니다.")