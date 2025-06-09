from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)
CORS(app)

model = tf.lite.Interpreter(model_path="mobilenet_v1_1.0_224.tflite")
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

with open("labels.txt") as f:
    labels = f.read().splitlines()

def preprocess(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).resize((224, 224)).convert('RGB')
    img = np.array(image).astype(np.float32)
    img = (img - 127.5) / 127.5
    return np.expand_dims(img, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = base64.b64decode(data["image"])
    input_data = preprocess(image_data)
    model.set_tensor(input_details[0]["index"], input_data)
    model.invoke()
    output_data = model.get_tensor(output_details[0]["index"])[0]
    top_indices = output_data.argsort()[-3:][::-1]
    return jsonify({
        "top1": f"{labels[top_indices[0]]}: {output_data[top_indices[0]]:.2f}",
        "top2": f"{labels[top_indices[1]]}: {output_data[top_indices[1]]:.2f}",
        "top3": f"{labels[top_indices[2]]}: {output_data[top_indices[2]]:.2f}",
    })

if __name__ == "__main__":
    app.run()
