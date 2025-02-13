import tensorflow as tf
import numpy as np
import cv2

class ASLClassifier:
    def __init__(self, model_path='asl_model_v1.h5', labels_path='class_labels_v1.npy'):
        self.model = tf.keras.models.load_model(model_path)
        data = np.load(labels_path, allow_pickle=True)
        self.class_labels = {i: label for i, label in enumerate(data)}

    def predict_asl(self, image_path: str = None, image_data: bytes = None):
        if image_path is None and image_data is None:
            raise ValueError("Either 'image_path' or 'image_data' must be provided.")

        if image_path is not None:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Unable to read image at path: {image_path}")
        else:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unable to decode image from provided raw data.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = self.model.predict(img)
        predicted_class_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_label = self.class_labels[predicted_class_idx]
        return predicted_label, confidence
