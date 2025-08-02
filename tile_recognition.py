from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ✅ Force Torch to ignore CUDA
app = Flask(__name__)
model = YOLO("best.pt")
model.to("cpu")  # ✅ force CPU mode


@app.route("/detect", methods=["POST"])
def detect_tiles():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        img_height, img_width = image.shape[:2]
        results = model.predict(image)[0]
        boxes = results.boxes.xywh.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        y_centers = boxes[:, 1]
        if np.max(y_centers) <= 1.0:
            boxes[:, [0, 2]] *= img_width
            boxes[:, [1, 3]] *= img_height

        y_centers = boxes[:, 1]
        tile_heights = boxes[:, 3]
        avg_tile_height = np.mean(tile_heights)
        lowest_y = max(y_centers)
        threshold = lowest_y - (1.05 * avg_tile_height)

        closed_indices = [i for i, y in enumerate(y_centers) if y > threshold]
        open_indices = [i for i in range(len(boxes)) if i not in closed_indices]

        tile_data = []
        for i, (x, y, w, h) in enumerate(boxes):
            tile_data.append({
                "label": results.names[classes[i]],
                "x_center": float(x),
                "y_center": float(y),
                "width": float(w),
                "height": float(h),
                "status": "closed" if i in closed_indices else "open"
            })

        return jsonify(tile_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
    return "Mahjong AI Flask backend is running!"

