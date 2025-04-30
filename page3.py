import streamlit as st
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
st.set_page_config(page_title="page3", page_icon="🔮")
st.write("""
# 🔮 Xác định điểm bất thường trong tiêu thụ điện năng
""")
@st.cache_resource
def load_keras_model(path):
    try:
        model = keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Không thể tải mô hình từ file {path}.")
        st.stop()

#tải mô hình keras
model3 = load_keras_model("best_model_mul_cnn16_lstm8_dense50_dense6.keras")    

st.title("⚠️ Anomaly Detection bằng Banpei SST")

uploaded = st.file_uploader("Tải lên CSV chứa cột 'value' (dãy thời gian)", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    series = df["value"].to_numpy()

    # 2) Tính errors, threshold động, và mask
    errors, threshold, mask = detect_anomalies(series, model3, k=3.0)
    df["Reconstruction_Error"] = errors
    df["Anomaly"] = mask

    # 3) Vẽ biểu đồ error + threshold
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(errors, label="Reconstruction Error")
    ax1.axhline(threshold, color="red", linestyle="--",
                label=f"Threshold = mean+3·std ({threshold:.5f})")
    ax1.legend()
    st.pyplot(fig1)

    # 4) Vẽ biểu đồ series, highlight anomaly
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(series, label="Value")
    ax2.scatter(np.where(mask)[0], series[mask],
                color="red", label="Anomaly", zorder=5)
    ax2.legend()
    st.pyplot(fig2)

    # 5) Bảng chi tiết anomalies
    st.subheader("🔎 Danh sách điểm bất thường")
    st.dataframe(df[df["Anomaly"]])