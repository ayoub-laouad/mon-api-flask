from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.lite.Interpreter(model_path="mobilenet_v1_1.0_224.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224)).convert('RGB')
    img_array = np.array(image).astype(np.float32)
    img_array = (img_array - 127.5) / 127.5
    return np.expand_dims(img_array, axis=0)

@app.post("/api/predict")
async def predict(request: Request):
    data = await request.json()
    image_data = base64.b64decode(data['image'])
    input_tensor = preprocess(image_data)
    model.set_tensor(input_details[0]['index'], input_tensor)
    model.invoke()
    output = model.get_tensor(output_details[0]['index'])[0]
    top_indices = output.argsort()[-3:][::-1]
    return {
        "top1": f"{labels[top_indices[0]]}: {output[top_indices[0]]:.2f}",
        "top2": f"{labels[top_indices[1]]}: {output[top_indices[1]]:.2f}",
        "top3": f"{labels[top_indices[2]]}: {output[top_indices[2]]:.2f}",
    }
