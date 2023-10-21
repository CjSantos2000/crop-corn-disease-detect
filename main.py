import base64
import io
from flask import Flask, request, jsonify, abort
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

model_rice = YOLO("rice_best_70.pt")
model_rice_names = model_rice.names

model_corn = YOLO("corn_best_80.pt")
model_corn_names = model_corn.names


@app.route("/")
def hello():
    return "Hello World!12345"


@app.route("/crop/detect-disease/v1/", methods=["POST"])
def detect_crop_disease_image():
    if "image" not in request.files:
        abort(400, "No image part")

    image_file = request.files["image"]

    # Check if the file has a valid content type (e.g., image/jpeg, image/png, >
    if not allowed_file(image_file.filename):
        abort(415, "Unsupported Media Type")

    # Read the image data and save it to a file
    image_data = image_file.read()
    img = Image.open(io.BytesIO(image_data))

    results = model_rice.predict(imgsz=512, source=img)
    result = None
    detected = []
    for r in results:
        for c in r.boxes.cls:
            detected.append(model_rice_names[int(c)].replace("_", " ").title())

    if detected:
        result = detected[0]

    print(detected)

    result_dict = {"output": "output_key", "detected": result, "type": "rice"}
    return jsonify(result_dict), 200


@app.route("/corn/detect-disease/v1/", methods=["POST"])
def detect_corn_disease_image():
    if "image" not in request.files:
        abort(400, "No image part")

    image_file = request.files["image"]

    # Check if the file has a valid content type (e.g., image/jpeg, image/png, >
    if not allowed_file(image_file.filename):
        abort(415, "Unsupported Media Type")

    # Read the image data and save it to a file
    image_data = image_file.read()
    img = Image.open(io.BytesIO(image_data))

    results = model_corn.predict(imgsz=512, source=img, conf=0.3)
    result = None
    detected = []
    for r in results:
        for c in r.boxes.cls:
            detected.append(model_corn_names[int(c)].replace("_", " ").title())

    if detected:
        result = detected[0]

    result_dict = {"output": "output_key", "detected": result, "type": "corn"}
    return jsonify(result_dict), 200


# Define a function to check if the file extension is allowed
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run()
