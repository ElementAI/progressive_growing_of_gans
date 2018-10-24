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
    Gs = model
    # Generate latent vectors.
    latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
    latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

    # Run the generator to produce a set of images.
    images = Gs.run(latents, labels)

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

    # Save images as PNG.
    for idx in range(images.shape[0]):
        PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)


# @api.route("/predict", methods=['POST'])
# def predict():

#     data = request.get_json()
#     if data is None:
#         print("no data")
#         abort(404)
#     if 'data' not in data:
#         print("data not in data")
#         abort(404)
#     else:
#         data = data['data']

#     data = np.array([data])
#     print(model)
#     print(data)
#     print(data.shape)
#     print(model.input_shapes)
#     if data.shape[1] != model.input_shapes[0][1]:
#         abort(403)
#     labels = np.zeros([data.shape[0]] + model.input_shapes[1][1:])

#     print(labels.shape)
#     images = model.run(data, labels)

#     # Convert array to Image
#     img = PIL.Image.fromarray(images[0])
#     return send_file(
#         io.BytesIO(img),
#         mimetype='image/png',
#         as_attachment=True,
#         attachment_filename='prediction.png'), 200


if __name__ == '__main__':
    api.run(host='0.0.0.0')
