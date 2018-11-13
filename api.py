import io
import os
import pickle

from flask import abort
from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file
from flask_cors import CORS
import numpy as np
import PIL
import tensorflow as tf
from os import listdir
from os.path import isfile, join

api = Flask(__name__)
CORS(api)


def load_model(path):
    with open(path, 'rb') as file:
        G, D, Gs = pickle.load(file)
    return G, D, Gs


# Initialize TensorFlow session.
SESS = tf.Session()

with SESS.as_default():
    with SESS.graph.as_default():
        _, _, model = load_model(os.environ.get("MODEL_PATH"))


@api.route("/models")
def models():
    models_path = os.environ.get("MODELS_PATH")
    models = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    models = [m.replace('.pkl', '') for m in models]
    return jsonify({'available_models': models}), 200


@api.route("/models/<name>")
def swap_models(name):
    global SESS
    global model
    models_path = os.environ.get("MODELS_PATH")
    models = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    file_name = name + '.pkl'
    data = np.array([np.random.normal(0, 1, size=512)])
    labels = np.zeros([data.shape[0]] + model.input_shapes[1][1:])
    if file_name in models:
        try:
            new_sess = tf.Session()
            with new_sess.as_default():
                with new_sess.graph.as_default():
                    _, _, model = load_model(
                        os.path.join(models_path, file_name))
                    images = model.run(data, labels)
                    print(images.shape)
            SESS.close()
            SESS = new_sess
        except Exception as e:
            print(e)
            print(file_name)
            return jsonify({'error': 'Fail to load the model'}), 403

    return jsonify({'loaded': file_name}), 200


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
        print("no data")
        abort(404)
    if 'data' not in data:
        print("data not in data")
        abort(404)
    else:
        data = data['data']

    data = np.array([data])
    print(data.shape)
    print(model.input_shapes)
    if data.shape[1] != model.input_shapes[0][1]:
        abort(403)
    labels = np.zeros([data.shape[0]] + model.input_shapes[1][1:])

    print(labels.shape)
    with SESS.as_default():
        with SESS.graph.as_default():
            images = model.run(data, labels)
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(
        np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)
    print(images.shape)
    # Convert array to Image
    img = PIL.Image.fromarray(images[0])
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')
    img_io.seek(0)
    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='prediction.png'), 200


if __name__ == '__main__':
    api.run(host='0.0.0.0')
