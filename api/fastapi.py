from fastapi import FastAPI, File, UploadFile
from scripts.infer import infer
import os

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open("temp.jpg", "wb") as f:
        f.write(await file.read())
    
    detections, cost = infer("temp.jpg")
    return {"detections": detections, "estimated_cost": cost}