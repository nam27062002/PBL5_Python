import cv2
import numpy as np
import tensorflow as tf

# Load model và class labels
model = tf.keras.models.load_model('asl_model.h5')
class_labels = list(model.train_generator.class_indices.keys())  # Lấy tên các lớp

# Đọc ảnh và tiền xử lý
image_path = 'dataset/asl_dataset/train/E/E1.jpg'  # Thay bằng đường dẫn ảnh của bạn
image = cv2.imread(image_path)
resized = cv2.resize(image, (224, 224))
normalized = resized / 255.0
input_img = np.expand_dims(normalized, axis=0)

# Dự đoán
predictions = model.predict(input_img)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)
label = f"Predicted: {class_labels[predicted_class]} ({confidence:.2f})"

# Hiển thị kết quả
cv2.putText(image, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('ASL Prediction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()