import os
import io
from PIL import Image
from ultralytics import YOLO
from flask import Flask, request, jsonify, abort

app = Flask(__name__)

model_rice = YOLO("rice_best_70.pt")
model_rice_names = model_rice.names

model_corn = YOLO("corn_best_50.pt")
model_corn_names = model_corn.names


@app.route("/corn/detect-disease/", methods=["POST"])
def detect_corn_disease():
    print(request.json)
    if not request.json or "image" not in request.json:
        abort(400)

    # get the base64 encoded string
    im_b64 = request.json["image"]

    # convert it into bytes
    img_bytes = base64.b64decode(im_b64.encode("utf-8"))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    img.save("Detect/test_corn.png")

    results = model_corn.predict(
        imgsz=512, source=img, save=False, save_txt=False, conf=0.3
    )
    detected = []
    for r in results:
        for c in r.boxes.cls:
            detected.append(model_corn_names[int(c)])

    result_dict = {"output": "output_key", "detected": detected[0]}
    return jsonify(result_dict)


@app.route("/crop/detect-disease/", methods=["POST"])
def detect_crop_disease():
    # print(request.json)
    if not request.json or "image" not in request.json:
        abort(400)

    # get the base64 encoded string
    im_b64 = request.json["image"]

    # convert it into bytes
    img_bytes = base64.b64decode(im_b64.encode("utf-8"))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    img.save("Detect/test.png")
    results = model_rice.predict(
        imgsz=512, source=img, save=False, save_txt=False, conf=0.6
    )
    detected = []
    for r in results:
        for c in r.boxes.cls:
            detected.append(model_rice_names[int(c)])

    result_dict = {"output": "output_key", "detected": detected[0]}
    return jsonify(result_dict)


# --------------------------------------------#


# @app.route("/corn/detect-disease/", methods=["POST"])
# def detect_corn_disease_image():
#     if "image" not in request.files:
#         abort(400, "No image part")

#     image_file = request.files["image"]

#     # Check if the file has a valid content type (e.g., image/jpeg, image/png, etc.)
#     if not allowed_file(image_file.filename):
#         abort(415, "Unsupported Media Type")

#     # Read the image data and save it to a file
#     image_data = image_file.read()
#     img = Image.open(io.BytesIO(image_data))
#     img.save("uploaded_image.png")

#     results = model_corn.predict(imgsz=512, source=img, conf=0.3)
#     detected = []
#     for r in results:
#         for c in r.boxes.cls:
#             detected.append(model_corn_names[int(c)])

#     result_dict = {"output": "output_key", "detected": detected[0]}
#     return jsonify(result_dict), 200


# @app.route("/crop/detect-disease/", methods=["POST"])
# def detect_crop_disease_image():
#     if "image" not in request.files:
#         abort(400, "No image part")

#     image_file = request.files["image"]

#     # Check if the file has a valid content type (e.g., image/jpeg, image/png, etc.)
#     if not allowed_file(image_file.filename):
#         abort(415, "Unsupported Media Type")

#     # Read the image data and save it to a file
#     image_data = image_file.read()
#     img = Image.open(io.BytesIO(image_data))
#     img.save("uploaded_image.png")

#     results = model_rice.predict(imgsz=512, source=img, conf=0.6)
#     detected = []
#     for r in results:
#         for c in r.boxes.cls:
#             detected.append(model_rice_names[int(c)])

#     result_dict = {"output": "output_key", "detected": detected[0]}
#     return jsonify(result_dict), 200


# # Define a function to check if the file extension is allowed
# def allowed_file(filename):
#     ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
