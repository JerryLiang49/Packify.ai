# Import necessary libraries
from ultralytics import YOLO
from roboflow import Roboflow
import os
import requests

# Download the YOLOv8s model if it does not exist
model_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"
model_path = os.path.expanduser("~/yolov8s.pt")

if not os.path.exists(model_path):
    print(f"Downloading {model_url} to {model_path}...")
    response = requests.get(model_url, allow_redirects=True)
    with open(model_path, 'wb') as f:
        f.write(response.content)

# Initialize Roboflow and download the dataset
rf = Roboflow(api_key="QFDXTpEE3QnIT4Nry4Dm")
project = rf.workspace("yolo-hpvvo").project("text-recognition2")
version = project.version(5)
dataset = version.download("yolov7")

# Define  
data_path = dataset.location + "/data.yaml"

# Verify paths
assert os.path.exists(model_path), f"Model file not found at {model_path}"
assert os.path.exists(data_path), f"Dataset config file not found at {data_path}"

# Initialize and train the model
model = YOLO(model_path)
model.train(data=data_path, epochs=300, plots=True)