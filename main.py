# --- Bước 1: Cài đặt thư viện ---
# Chạy các lệnh này trong terminal trước:
# pip install tensorflow[and-cuda] Pillow numpy matplotlib

# --- Bước 2: Import thư viện ---
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# --- Bước 3: Kiểm tra GPU ---
print("=" * 50)
print("Danh sách GPU có sẵn:", tf.config.list_physical_devices('GPU'))
print("=" * 50)

# --- Bước 4: Cấu hình GPU ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Giới hạn bộ nhớ GPU ở mức 4GB
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print("Đã cấu hình GPU thành công!")
    except RuntimeError as e:
        print(e)

# --- Bước 5: Tiền xử lý dữ liệu ---
# Đường dẫn dataset
train_dir = './dataset/asl_dataset/train'
test_dir = './dataset/asl_dataset/test'

# Data augmentation cho tập train
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    validation_split=0.2
)

# Generator cho tập train và validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # MobileNetV2 yêu cầu 224x224
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# --- Bước 6: Xây dựng model với Transfer Learning ---
# Sử dụng MobileNetV2 làm base model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Đóng băng các layer của base model
base_model.trainable = False

# Xây dựng model đầu ra
model = Sequential([
    base_model,
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(29, activation='softmax')  # 29 lớp đầu ra cho ASL
])

# Biên dịch model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Bước 7: Huấn luyện model ---
# Callbacks để tối ưu quá trình training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
]

# Training
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# --- Bước 8: Đánh giá model ---
# Vẽ đồ thị accuracy và loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()


# --- Bước 9: Dự đoán ảnh mới ---
def predict_asl(image_path):
    # Tiền xử lý ảnh
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Dự đoán
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_label = list(train_generator.class_indices.keys())[class_idx]

    return class_label


# Ví dụ sử dụng
test_image = './dataset/asl_dataset/test/A_test.jpg'  # Thay bằng đường dẫn ảnh của bạn
print("Dự đoán:", predict_asl(test_image))

# --- Bước 10: Lưu model ---
model.save('asl_model.h5')
print("Đã lưu model thành công!")