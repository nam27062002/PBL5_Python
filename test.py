import numpy as np

# Load file .npy
data = np.load('class_labels_v1.npy', allow_pickle=True)

# Kiểm tra nội dung file
print("Nội dung file .npy:", data)
print("Kiểu dữ liệu:", type(data))
print("Kích thước mảng:", data.shape)

# Chuyển đổi mảng thành từ điển
class_labels = {i: label for i, label in enumerate(data)}

# In kết quả để kiểm tra
print("Class Labels (dictionary):", class_labels)