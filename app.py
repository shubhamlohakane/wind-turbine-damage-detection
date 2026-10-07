import os
from pathlib import Path

import cv2
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
MODEL_PATH = BASE_DIR / "models" / "best.pt"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
CLASS_NAMES = ["LE Erosion", "PU Tape Damage", "Paint Peeloff"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
model = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global model
    if model is not None:
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file is missing. Add models/best.pt to the project before deployment."
        )

    # Downloads the YOLOv5 code automatically on first startup, so the repository
    # does not need to be committed into GitHub.
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=str(MODEL_PATH),
        force_reload=False,
        trust_repo=True,
    )
    model.conf = 0.25
    model.iou = 0.45


def predict(image_bgr, confidence=0.25):
    load_model()
    model.conf = confidence
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)

    detections = []
    rows = results.xyxy[0].detach().cpu().numpy()
    for x1, y1, x2, y2, conf, cls_id in rows:
        cls_id = int(cls_id)
        name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
        detections.append({
            "class": name,
            "confidence": round(float(conf) * 100, 2),
            "box": {
                "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                "x2": round(float(x2), 1), "y2": round(float(y2), 1),
            },
        })

    results.render()
    rendered_rgb = results.ims[0]
    rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
    return rendered_bgr, detections


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "model_exists": MODEL_PATH.exists()})


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "Please select an image."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Please select an image."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Use JPG, JPEG, PNG or WEBP."}), 400

    try:
        confidence = float(request.form.get("confidence", "0.25"))
        confidence = max(0.05, min(confidence, 0.95))
    except ValueError:
        confidence = 0.25

    safe_name = secure_filename(file.filename)
    input_path = UPLOAD_DIR / safe_name
    file.save(input_path)
    image = cv2.imread(str(input_path))
    if image is None:
        return jsonify({"error": "Could not read the uploaded image."}), 400

    try:
        rendered, detections = predict(image, confidence)
    except Exception as exc:
        app.logger.exception("Inference failed")
        return jsonify({"error": f"Model inference failed: {exc}"}), 500

    result_name = f"{Path(safe_name).stem}_result.jpg"
    cv2.imwrite(str(RESULT_DIR / result_name), rendered)
    counts = {name: 0 for name in CLASS_NAMES}
    for item in detections:
        counts[item["class"]] = counts.get(item["class"], 0) + 1

    return jsonify({
        "success": True,
        "result_url": f"/results/{result_name}",
        "detections": detections,
        "total": len(detections),
        "counts": counts,
    })


@app.route("/results/<path:filename>")
def results(filename):
    return send_from_directory(RESULT_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
