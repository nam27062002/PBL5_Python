import random
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import cv2
from tensorflow.keras.models import load_model

# Đường dẫn đến mô hình và thư mục train
MODEL_PATH = 'asl_mobile_net_v2_model.h5'  # Đường dẫn tới file mô hình của bạn
TRAIN_DIR = Path('../dataset/asl_dataset/train')  # Đường dẫn đến thư mục train

# Tải mô hình
model = load_model(MODEL_PATH)

# Lấy danh sách tất cả các thư mục con (lớp)
class_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]

# Tạo danh sách tất cả ảnh và nhãn tương ứng
all_images = []
all_labels = []
for class_dir in class_dirs:
    class_name = class_dir.name
    images = list(class_dir.glob('*.jpg'))  # Giả sử ảnh định dạng .jpg
    all_images.extend(images)
    all_labels.extend([class_name] * len(images))

# Chọn ngẫu nhiên 1000 ảnh
random.seed(42)  # Đảm bảo kết quả có thể tái lập
sample_indices = random.sample(range(len(all_images)), 1000)
sample_images = [all_images[i] for i in sample_indices]
sample_labels = [all_labels[i] for i in sample_indices]

# Tạo ánh xạ từ tên lớp sang chỉ số
class_names = sorted(set(all_labels))
class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}

# Hàm tiền xử lý ảnh
def load_and_preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize về kích thước 224x224
    img = img / 255.0  # Chuẩn hóa giá trị pixel về [0, 1]
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch
    return img

# Hàm dự đoán và đo thời gian
def predict_and_time(model, image_path):
    start_time = time.time()
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    end_time = time.time()
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class, end_time - start_time

# Dự đoán và ghi nhận kết quả với thanh tiến trình
predicted_classes = []
inference_times = []
true_indices = [class_to_index[label] for label in sample_labels]

for img_path in tqdm(sample_images, desc="Đang dự đoán"):
    pred_class, inf_time = predict_and_time(model, img_path)
    predicted_classes.append(pred_class)
    inference_times.append(inf_time)

# Tính độ chính xác trung bình
accuracy = accuracy_score(true_indices, predicted_classes)

# Tính thời gian suy luận trung bình
average_time = np.mean(inference_times)

# Hiển thị kết quả
print(f"\nĐộ chính xác trung bình trên 1000 ảnh: {accuracy * 100:.2f}%")
print(f"Thời gian suy luận trung bình: {average_time:.4f} giây")