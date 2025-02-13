import requests
import json

url = "http://localhost:8000/predict"
file_path = "./dataset/a.jpg"

try:
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "image/jpeg")}
        response = requests.post(url, files=files)

    result = response.json()
    if result["status"] == "success":
        print(f"Kết quả: {result['predicted_class']} (Độ tin cậy: {result['confidence'] * 100:.2f}%)")
    else:
        print(f"Lỗi: {result.get('error', 'Unknown error')}")

except Exception as e:
    print(f"Lỗi kết nối: {str(e)}")