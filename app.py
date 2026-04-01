# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 1. 데이터와 모델 로드
# ==========================
# df_dong_merged와 모델을 pickle로 저장해두었다고 가정
import pickle

with open('df_dong_merged.pkl', 'rb') as f:
    df_dong_merged = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="서울 카페 상권 위험 분석", layout="wide")
st.title("서울시 카페 상권 폐업률 예측 대시보드")

# ==========================
# 2. 개별 행정동 위험 예측
# ==========================
st.header("개별 행정동 예측")
gu_selected = st.selectbox("자치구 선택", df_dong_merged['시군구명'].unique())
dong_selected = st.selectbox(
    "행정동 선택", 
    df_dong_merged[df_dong_merged['시군구명']==gu_selected]['행정동명'].unique()
)

# 선택한 동 데이터
dong_data = df_dong_merged[
    (df_dong_merged['시군구명']==gu_selected) &
    (df_dong_merged['행정동명']==dong_selected)
]

st.subheader("예측 폐업률 & 위험등급")
pred_rate = model.predict(dong_data[[
    '집객시설_수','카페_수','카페밀도지수',
    '유사업종비율','폐업밀도','지하철_역_수','버스_정거장_수'
]])[0]

risk_label = dong_data['위험등급'].values[0]

st.write(f"예측 폐업률: {pred_rate:.2f}%")
st.write(f"위험등급: {risk_label}")

# ==========================
# 3. 상위 위험 상권
# ==========================
st.header("고위험 상권 Top 10")
top_risk = df_dong_merged.sort_values(by='예측_폐업률', ascending=False).head(10)
st.dataframe(top_risk[['시군구명','행정동명','예측_폐업률','위험등급','카페_수','집객시설_수']])

# ==========================
# 4. 시각화
# ==========================
st.header("상권 분석 시각화")
st.subheader("카페밀도 vs 폐업률")
plt.figure(figsize=(10,5))
sns.scatterplot(
    data=df_dong_merged,
    x='카페밀도지수',
    y='폐업_률',
    hue='위험등급',
    palette={'안전':'green','주의':'orange','위험':'red'}
)
plt.xlabel('카페밀도지수')
plt.ylabel('실제 폐업률')
plt.title('카페밀도 vs 폐업률')
st.pyplot(plt)

st.subheader("집객시설 수별 예측 폐업률")
plt.figure(figsize=(10,5))
sns.scatterplot(
    data=df_dong_merged,
    x='집객시설_수',
    y='예측_폐업률',
    hue='위험등급',
    palette={'안전':'green','주의':'orange','위험':'red'}
)
plt.xlabel('집객시설 수')
plt.ylabel('예측 폐업률')
plt.title('집객시설 수 vs 예측 폐업률')
st.pyplot(plt)