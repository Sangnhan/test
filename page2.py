import streamlit as st
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
st.set_page_config(page_title="page2", page_icon="🔮")

# ---------------------------
# 1. Hàm tải mô hình với caching
# ---------------------------
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
st.title("🔮 DỰ BÁO CHỈ SỐ CÔNG TƠ ĐIỆN")

st.markdown("""
Nhập các thông số đầu vào sau đây:
- **Ngày**: Số từ 1 đến 31
- **Tháng**: Số từ 1 đến 12
- **Năm**: Ví dụ: 2023
- **Quý**: Số từ 1 đến 4
- **Giờ**: Số từ 0 đến 23
""")

Timestamp = st.number_input("Ngày:", min_value=1, max_value=31, value=1)
Month = st.number_input("Tháng:", min_value=1, max_value=12, value=3)
#Quarter = st.number_input("Quý:", min_value=1, max_value=4, value=1)
Year = st.number_input("Năm:", min_value=2000, max_value=2100, value=2025)
selected_hour = st.selectbox("Giờ:", ["0", "6", "12", "18"])
hour_map = {"0": 1, "6": 2, "12": 3, "18": 4}
Hour = hour_map[selected_hour]

# Tạo mảng đầu vào có shape (1,4)
input_features = np.array([[Timestamp, Month,Year, Hour]])

st.markdown("### KẾT QUẢ:")

# ---------------------------
# 4. Hiển thị các nút dự đoán cho từng mô hình
# ---------------------------
if st.button("Dự đoán"):
    try:
        hours = ["0h", "6h", "12h", "18h"]
        hour_indices = [1, 2, 3, 4]  # ánh xạ giờ thành chỉ số

        predicted_values = []

        for hour in hour_indices:
            input_features = np.array([[Timestamp, Month, Year, hour]])
            sequence = np.repeat(input_features, 144, axis=0)  # (144, 4)
            input_feature = np.expand_dims(sequence, axis=0)   # (1, 144, 4)

            prediction = selected_model.predict(input_feature)
            predicted = prediction[0][0] if prediction.ndim > 1 else prediction[0]
            predicted_values.append(predicted)

        # Hiển thị giá trị dự đoán
        for i in range(len(hours)):
            st.write(f"🔹 {hours[i]}: {predicted_values[i]:.2f}")

        # Vẽ biểu đồ cột
        fig, ax = plt.subplots()
        ax.bar(hours, predicted_values, color='skyblue')
        ax.set_title("Dự đoán chỉ số công tơ điện trong ngày")
        ax.set_xlabel("Thời gian trong ngày")
        ax.set_ylabel("Chỉ số công tơ điện dự đoán")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")