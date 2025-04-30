import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

st.set_page_config(page_title="page4", page_icon="⚡")
@st.cache_resource
def load_keras_model(path):
    try:
        model = keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Không thể tải mô hình từ file {path}.")
        st.stop()

# ---------------------------
# 2. Tải 4 mô hình Keras
# ---------------------------
#model1 = load_keras_model("best_model_Conv16_lstm32_lstm16_dense64.keras")
#model2 = load_keras_model("best_model_mul_cnn16_lstm8_dense50_dense6 copy.keras")
model3 = load_keras_model("best_model_mul_cnn16_lstm8_dense50_dense6.keras")
# ---------------------------
# 3. Giao diện nhập liệu
# ---------------------------
def user_input_features():
    selected_model = st.sidebar.selectbox("Chọn mô hình dự đoán",["CNN"])
    return selected_model
# Lấy tên mô hình được chọn từ giao diện
selected_model_name = user_input_features()
# Hàm ánh xạ mô tên mô hình với đối tượng tương ứng 
def get_prediction_model(model_name):
    mapping = {
     #   "LSTM" : model1,
     #   "BI_LSTM" : model2,
        "CNN" : model3,
    }
    
    
    if model_name in mapping:
        return mapping[model_name]
    else:
        st.error("Mô hình không hợp lệ")
        st.stop()
#Lấy mô hình dự đoán tương ứng với lựa chọn
selected_model = get_prediction_model(selected_model_name)






st.title("📈 DỰ BÁO CHỈ SỐ CÔNG TƠ 7 NGÀY TIẾP THEO")

uploaded_file = st.file_uploader("📄 Tải lên file CSV (có cột: Date, Hour, Value)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Date", "Hour"])

        st.markdown("### 📅 Chọn ngày bắt đầu (lấy 7 ngày tính từ ngày này)")
        unique_dates = df["Date"].dt.date.unique()
        selected_start_date = st.selectbox("Ngày bắt đầu", unique_dates[::-1])  # Mới nhất trước

        # Lọc 7 ngày liên tiếp
        selected_range = pd.date_range(start=selected_start_date, periods=7)
        df_selected = df[df["Date"].dt.date.isin(selected_range.date)]

        st.markdown("### 📋 Dữ liệu của 7 ngày được chọn:")
        st.dataframe(df_selected)   

        if df_selected.shape[0] != 28:
            st.warning("⚠️ Thiếu dữ liệu — cần đủ 7 ngày x 4 lần đo (tổng cộng 28 dòng).")
            st.dataframe(df_selected)
        else:
            input_seq = df_selected["Value"].values.reshape(1, 28, 1)

            if st.button("🚀 Dự đoán"):
                prediction = selected_model.predict(input_seq).flatten()  # (28,)
                
                st.markdown("## 📊 Dự đoán 7 ngày tiếp theo")
                fig, ax = plt.subplots()

                for i in range(7):
                    values = prediction[i*4:(i+1)*4]
                    ax.plot(["0h", "6h", "12h", "18h"], values, marker="o", label=f"Ngày {i+1}")

                ax.set_title("Dự đoán chỉ số công tơ 7 ngày tới")
                ax.set_xlabel("Giờ")
                ax.set_ylabel("Chỉ số công tơ")
                ax.legend()
                st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Lỗi xử lý dữ liệu: {e}")
