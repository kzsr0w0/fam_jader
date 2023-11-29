import streamlit as st
import pandas as pd
import requests

st.title('Famotidine SE Fearures')

st.sidebar.header('Input Features')
gender = st.sidebar.slider('gender', min_value=0, max_value=2, step=1)
age = st.sidebar.slider('age', min_value=9, max_value=100, step=1)
bw = st.sidebar.slider('body weight', min_value=9, max_value=170, step=1)
height = st.sidebar.slider('height', min_value=9, max_value=180, step=1)

jader = {
    'gender': gender,
    'age': age,
    'bw': bw,
    'height': height
}



if st.sidebar.button("Predict"):
    # 入力された説明変数の表示
    st.write('## Input Value')
    jader_df = pd.DataFrame(jader, index=['data'])
    st.write(jader_df)

    # 予測の実行
    response = requests.post("http://127.0.0.1:8000/make_predictions", json=jader_df)
    prediction = response.json()["prediction"]

    # 予測結果の表示
    st.write('## Prediction')
    st.write(prediction)