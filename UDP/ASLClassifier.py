import tensorflow as tf
import numpy as np
import cv2


class ASLClassifier:
    def __init__(self, model_path: str, class_labels: list = None, input_size: tuple = (224, 224)):
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise ValueError(f"Không thể tải mô hình từ {model_path}: {str(e)}")

        self.input_size = input_size

        if class_labels is None:
            self.class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
                                 "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
        else:
            self.class_labels = class_labels

        output_shape = self.model.output_shape[-1]
        if output_shape != len(self.class_labels):
            raise ValueError(
                f"Số lớp trong mô hình ({output_shape}) không khớp với số nhãn lớp ({len(self.class_labels)}).")

    def predict_asl(self, image_path: str = None, image_data: bytes = None, model_type: str = "mobilenetv2"):
        if image_path is None and image_data is None:
            raise ValueError("Phải cung cấp ít nhất một trong hai: 'image_path' hoặc 'image_data'.")

        if image_path is not None:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        else:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Không thể giải mã ảnh từ dữ liệu byte thô.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)

        if model_type.lower() == "vgg16":
            img = img - [123.68, 116.779, 103.939]
        else:
            img = img / 255.0

        img = np.expand_dims(img, axis=0)

        predictions = self.model.predict(img)
        predicted_class_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_label = self.class_labels[predicted_class_idx]
        return predicted_label, confidence
