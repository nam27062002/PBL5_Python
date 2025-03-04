import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Đường dẫn tới mô hình và ảnh
MODEL_PATH = 'asl_mobile_net_v2_model.h5'  # Đường dẫn tới file mô hình của bạn
IMAGE_PATH = 'D:\\PBL55555\\dataset\\K.jpg'  # Đường dẫn tới ảnh cần dự đoán

# Tải mô hình
model = load_model(MODEL_PATH)

# Hàm tải và xử lý ảnh
def load_and_preprocess_image(image_path):
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path)
    # Chuyển từ BGR (OpenCV) sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Thay đổi kích thước ảnh về 224x224 (kích thước mà MobileNetV2 thường yêu cầu)
    img = cv2.resize(img, (224, 224))
    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    img = img / 255.0
    # Thêm chiều batch để phù hợp với input của mô hình
    img = np.expand_dims(img, axis=0)
    return img

# Tải và xử lý ảnh
test_image = load_and_preprocess_image(IMAGE_PATH)

# Dự đoán nhãn
prediction = model.predict(test_image)
predicted_class = np.argmax(prediction, axis=1)[0]  # Lấy chỉ số lớp có xác suất cao nhất
confidence = np.max(prediction)  # Xác suất cao nhất

# Danh sách nhãn (cần thay đổi theo danh sách thực tế của bạn)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
predicted_label = class_names[predicted_class]

# In kết quả
print(f'Nhãn dự đoán: {predicted_label}')
print(f'Độ tin cậy: {confidence:.2%}')

# Hiển thị ảnh kèm kết quả
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(f'Dự đoán: {predicted_label} ({confidence:.2%})')
plt.axis('off')
plt.show()