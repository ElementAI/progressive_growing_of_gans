import io
import os
import pickle


from flask import abort
from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file
import numpy as np
import PIL
import tensorflow as tf


api = Flask(__name__)


def load_model(path):
    with open(path, 'rb') as file:
        G, D, Gs = pickle.load(file)
    return G, D, Gs


# Initialize TensorFlow session.
tf.InteractiveSession()


_, _, model = load_model(os.environ.get("MODEL_PATH"))


@api.route("/healthcheck")
def healthcheck():
    return jsonify({'Status': 'All good'}), 200


@api.route("/config")
def config():
    return jsonify({'input_shape': model.input_shapes[0]})


@api.route("/predict", methods=['POST'])
def predict():

    data = request.get_json()
    if data is None:
        abort(404)
    if 'data' not in data:
        abort(404)
    else:
        data = data['data']

    data = np.array([data])
    if data.shape[1] != model.input_shape[0][1]:
        abort(403)
    labels = np.zeros([data.shape[0]] + model.input_shapes[1][1:])

    images = model.run(data, labels)

    # Convert array to Image
    img = PIL.Image.fromarray(images[0])
    return send_file(
        io.BytesIO(img),
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='prediction.png'), 200


if __name__ == '__main__':
    api.run(host='0.0.0.0', debug=True)
