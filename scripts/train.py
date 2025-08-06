from ultralytics import YOLO
import os

def train_yolo():
    model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 nano model
    data_yaml = """
    train: ../data/processed/train
    val: ../data/processed/val
    nc: 29  # Updated to match combined dataset (21 parts + 8 damages)
    names: ['Quarter-panel', 'Front-wheel', 'Back-window', 'Trunk', 'Front-door', 'Rocker-panel', 'Grille', 'Windshield', 'Front-window', 'Back-door', 'Headlight', 'Back-wheel', 'Back-windshield', 'Hood', 'Fender', 'Tail-light', 'License-plate', 'Front-bumper', 'Back-bumper', 'Mirror', 'Roof', 'Missing part', 'Broken part', 'Scratch', 'Cracked', 'Dent', 'Flaking', 'Paint chip', 'Corrosion']
    """
    
    with open('data.yaml', 'w') as f:
        f.write(data_yaml)
    
    # Using CPU instead of GPU as its Mac
    model.train(data='data.yaml', epochs=5, imgsz=640, batch=16, device='cpu')

if __name__ == "__main__":
    train_yolo()