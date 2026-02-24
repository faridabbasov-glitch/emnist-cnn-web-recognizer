from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import base64
import json
from PIL import Image
import io
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'best_model.h5'))

with open(os.path.join(BASE_DIR, 'label_map.json')) as f:
    LABEL_MAP = json.load(f)

class ImageData(BaseModel):
    image: str

def preprocess(base64_str: str) -> np.ndarray:
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('L') 
    
    arr = np.array(img, dtype=np.float32)

    if np.mean(arr) > 127: 
        arr = 255.0 - arr

    coords = np.argwhere(arr > 30) 
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        digit = Image.fromarray(arr[y_min:y_max+1, x_min:x_max+1])
        digit.thumbnail((20, 20), Image.LANCZOS)
        
        new_img = Image.new('L', (28, 28), 0)
        new_img.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
        arr = np.array(new_img, dtype=np.float32)
    else:
        arr = np.zeros((28, 28), dtype=np.float32)

    arr /= 255.0
    arr = np.flip(arr, axis=1) 

    return arr.reshape(1, 28, 28, 1)


@app.post('/predict')
async def predict(data: ImageData):
    try:
        img = preprocess(data.image)
        probs = model.predict(img)[0]
        probs = np.nan_to_num(probs, nan=0.0)  
        top3_idx = np.argsort(probs)[::-1][:3]
        results = [
            {
                "label": LABEL_MAP.get(str(i), str(i)),
                "confidence": round(float(probs[i]), 4)
            }
            for i in top3_idx
        ]
        return {"predictions": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

frontend_dir = os.path.join(BASE_DIR, 'frontend')
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
