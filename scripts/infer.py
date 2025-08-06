from ultralytics import YOLO
import cv2
import sqlite3
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

def estimate_cost(detections, db_path='database/parts_costs.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_cost = 0
    for det in detections:
        part, conf, severity = det['class'], det['confidence'], det.get('severity', 'moderate')
        cursor.execute("SELECT repair_cost, replace_cost FROM parts_costs WHERE part_name=?", (part,))
        result = cursor.fetchone()
        if result:
            cost = result[1] if severity == 'severe' else result[0]
            total_cost += cost
    
    conn.close()
    return total_cost

def infer(image_path, model_path='runs/detect/train/weights/best.pt'):
    model = YOLO(model_path)
    results = model.predict(image_path)
    
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'class': result.names[int(box.cls)],
                'confidence': float(box.conf),
                'severity': 'moderate'  # Placeholder; enhance with severity model
            })
    
    cost = estimate_cost(detections)
    return detections, cost

if __name__ == "__main__":
    image_path = 'data/processed/test.jpg'
    detections, cost = infer(image_path)
    print(f"Detections: {detections}")
    print(f"Estimated Cost: ${cost}")