from flask import Flask, request
from flask_cors import CORS, cross_origin
import tensorflow as tf
from tensorflow import keras
import numpy as np
import base64

model = tf.keras.models.load_model("../model")
class_names = ['attractive', 'not-attractive']
app = Flask(__name__)
cors = CORS(app)


@app.route("/", methods=["POST"])
@cross_origin()
def root():
    FILE_PATH = r"C:\Users\Tyler Kowalski\Desktop\flask.jpg"

    content = request.json
    img_b64 = content["img"]
    img_bytes = base64.b64decode(img_b64)
    with open(FILE_PATH, "wb") as f:
        f.write(img_bytes)

    path = r"C:\Users\Tyler Kowalski\Desktop\flask.png"
    test_image = keras.preprocessing.image.load_img(path)
    img_array = keras.preprocessing.image.img_to_array(test_image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    print(predictions)
    score = tf.nn.softmax(predictions[0])
    print(score)
    print(class_names[np.argmax(score)])
    print(np.max(score))

    if class_names[np.argmax(score)] == "attractive":
        return { "result": np.max(score).item() }
    else:
        return { "result": -1 * np.max(score).item() }
