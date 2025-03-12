import cv2
import mediapipe as mp
import numpy as np
import os
from config.config import *

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.05,
    min_tracking_confidence=0.5
)

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc hình ảnh: {image_path}")
        return False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        print(f"Không phát hiện tay trong hình ảnh: {image_path}")
        return False
    height, width, _ = image.shape
    hand_mask = np.zeros((height, width), dtype=np.uint8)
    for hand_landmarks in results.multi_hand_landmarks:
        points = []
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            points.append((x, y))
        cv2.fillPoly(hand_mask, [np.array(points, dtype=np.int32)], 255)
    kernel = np.ones((5, 5), np.uint8)
    hand_mask_dilated = cv2.dilate(hand_mask, kernel, iterations=1)
    hand_mask_blurred = cv2.GaussianBlur(hand_mask_dilated, (21, 21), 0)
    hand_on_black_background = np.where(hand_mask_blurred[..., None] > 0, image, np.zeros_like(image))
    contours, _ = cv2.findContours(hand_mask_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Không tìm thấy đường viền trong hình ảnh: {image_path}")
        return False
    x_min, x_max, y_min, y_max = width, 0, height, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)
    hand_area = hand_on_black_background[y_min:y_max, x_min:x_max]
    hand_width, hand_height = x_max - x_min, y_max - y_min
    target_size = (224, 224)
    occupy_percent = 0.8
    scale_factor = min(target_size[0] / hand_width, target_size[1] / hand_height) * occupy_percent
    scaled_w, scaled_h = int(hand_width * scale_factor), int(hand_height * scale_factor)
    resized_hand_area = cv2.resize(hand_area, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    centered_image = np.zeros(target_size + (3,), dtype=np.uint8)
    x_offset = (target_size[0] - scaled_w) // 2
    y_offset = (target_size[1] - scaled_h) // 2
    centered_image[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized_hand_area
    grayscale_image = cv2.cvtColor(centered_image, cv2.COLOR_RGB2GRAY)
    if cv2.imwrite(output_path, grayscale_image):
        print(f"Đã lưu file xử lý: {output_path}")
        return True
    else:
        print(f"Lưu file xử lý thất bại: {output_path}")
        return False

def process_directory(input_dir, output_dir):
    for label in LABELS:
        input_label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)
        if not os.path.exists(input_label_dir):
            print(f"Thư mục {input_label_dir} không tồn tại!")
            continue
        files = [f for f in os.listdir(input_label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for file in files:
            input_file_path = os.path.join(input_label_dir, file)
            output_file_path = os.path.join(output_label_dir, file)
            print(f"Đang xử lý: {input_file_path}")
            process_image(input_file_path, output_file_path)

process_directory("../dataset/raw/Train_Alphabet", "../dataset/processed/v1/training")
process_directory("../dataset/raw/Test_Alphabet", "../dataset/processed/v1/validation")