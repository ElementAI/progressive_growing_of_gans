import hashlib
import io
import json
import os
import pickle
from os import listdir
from os.path import isfile, join
from pathlib import Path

import requests

import boto3
import numpy as np
import PIL
import qrcode
import tensorflow as tf
import twitter
from flask import Flask, abort, jsonify, request, send_file
from flask_cors import CORS
from utils.config import Config

cache = False
cache_dir = "/tmp"

s3_bucket_name = ""
s3_directory = ""

app = Flask(__name__)
CORS(app)

SESS = None
model = None
model_name = None
twitter_api = None
bitly_access_token = None
google_api_key = None


def init():
    global SESS, model, model_name, cache, cache_dir
    global s3_bucket_name, s3_directory
    global twitter_api
    global bitly_access_token
    global google_api_access_token

    cache = Config.get('cache')
    cache_dir = Config.get('cache_dir')
    if cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    s3_bucket_name = Config.get('s3_bucket_name')
    s3_directory = Config.get('s3_directory')
    consumer_key = Config.get('consumer_key')
    consumer_secret = Config.get('consumer_secret')
    access_token = Config.get('access_token')
    access_token_secret = Config.get('access_token_secret')
    bitly_access_token = Config.get('bitly_access_token')
    google_api_access_token = Config.get('google_api_access_token')

    twitter_api = twitter.Api(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token_key=access_token,
        access_token_secret=access_token_secret)
    print(twitter_api.VerifyCredentials())
    access_token = os.getenv(bitly_access_token)
    # Initialize TensorFlow session.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    SESS = tf.Session(config=tf_config)
    with SESS.as_default():
        with SESS.graph.as_default():
            model_path = os.environ.get("MODEL_PATH")
            model_name = os.path.basename(model_path)
            _, _, model = load_model(model_path)
            data = np.array([np.random.normal(0, 1, size=512)])
            labels = np.zeros([data.shape[0]] + model.input_shapes[1][1:])
            images = model.run(data, labels)
            print(images.shape)


def shorten(uri):
    query_params = {'access_token': bitly_access_token, 'longUrl': uri}

    endpoint = 'https://api-ssl.bitly.com/v3/shorten'
    response = requests.get(endpoint, params=query_params, verify=False)

    data = response.json()

    if not data['status_code'] == 200:
        print("Unexpected status_code: {} in bitly response. {}".format(data[
            'status_code'], response.text))

    return data['data']['url']


def shorteng(long_url):
    data = json.dumps({'longUrl': long_url})
    google_url = "https://www.googleapis.com/urlshortener/v1/url?key={}".format(
        google_api_access_token)

    result = requests.post(
        google_url, headers={'content-type': 'application/json'}, data=data)
    short_url = result.json()
    return short_url


def shortent(long_url):
    # data = json.dumps({'longUrl': long_url})
    data = {'url': long_url}
    google_url = "http://tinyurl.com/api-create.php"

    result = requests.post(
        google_url, headers={'content-type': 'application/json'}, params=data)
    text_data = result.text
    print(text_data)
    return text_data


def load_model(path):
    with open(path, 'rb') as file:
        G, D, Gs = pickle.load(file)
    return G, D, Gs


@app.route("/models")
def models():
    models_path = os.environ.get("MODELS_PATH")
    models = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    models = [m.replace('.pkl', '') for m in models]
    return jsonify({'available_models': models}), 200


@app.route("/models/<name>")
def swap_models(name):
    global SESS, model, model_name
    models_path = os.environ.get("MODELS_PATH")
    models = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    file_name = name + '.pkl'
    data = np.array([np.random.normal(0, 1, size=512)])
    labels = np.zeros([data.shape[0]] + model.input_shapes[1][1:])
    if file_name in models:
        try:
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
            new_sess = tf.Session(config=tf_config)
            with new_sess.as_default():
                with new_sess.graph.as_default():
                    model_name = os.path.basename(file_name)
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


@app.route("/healthcheck")
def healthcheck():
    return jsonify({'Status': 'All good'}), 200


@app.route("/config")
def config():
    global model
    return jsonify({'input_shape': model.input_shapes[0]})


def get_prediction(data):
    global SESS, model, model_name

    cache_file = None
    if cache:
        hash_str = "_".join([str(d) for d in data]).encode()
        cache_file = os.path.join(
            cache_dir,
            hashlib.sha256(hash_str).hexdigest() + '_' + model_name + '.png')
        if os.path.exists(cache_file):
            return cache_file

    data = np.array([data])
    # print(data.shape)
    # print(model.input_shapes)
    if data.shape[1] != model.input_shapes[0][1]:
        abort(403)
    labels = np.zeros([data.shape[0]] + model.input_shapes[1][1:])

    # print(labels.shape)
    with SESS.as_default():
        with SESS.graph.as_default():
            images = model.run(data, labels)
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(
        np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)
    # print(images.shape)
    # Convert array to Image
    img = PIL.Image.fromarray(images[0])
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')
    if cache and cache_file:
        img.save(cache_file, format='PNG')
    img_io.seek(0)
    return img_io


@app.route("/predict", methods=['POST'])
def predict():
    global SESS, model, model_name

    data = request.get_json()
    if data is None:
        print("no data")
        abort(404)
    if 'data' not in data:
        print("data not in data")
        abort(404)
    else:
        data = data['data']

    prediction = get_prediction(data)

    return send_file(
        prediction,
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='prediction.png'), 200


@app.route("/upload", methods=['POST'])
def upload_s3():
    global s3_bucket_name, s3_directory

    if not s3_bucket_name:
        raise 'Empty s3 bucket name ! Set env var S3_BUCKET_NAME=...'

    data_json = request.get_json()
    if data_json is None:
        print("no json data")
        return "no json data", 400
    if 'data' not in data_json:
        print("data not in json data")
        return "data not in data", 400

    data = data_json['data']

    hash_str = "_".join([str(d) for d in data]).encode()
    filename = hashlib.sha256(hash_str).hexdigest() + '.png'
    if s3_directory:
        filename = os.path.join(s3_directory, filename)

    prediction = get_prediction(data)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(s3_bucket_name)
    if isinstance(prediction, str):
        with open(prediction, 'rb') as fp:
            bucket.put_object(Key=filename, Body=fp, ACL='public-read')
    else:
        bucket.put_object(Key=filename, Body=prediction, ACL='public-read')

    bucket_location = boto3.client('s3').get_bucket_location(
        Bucket=s3_bucket_name)
    object_url = "https://s3-{0}.amazonaws.com/{1}/{2}".format(
        bucket_location['LocationConstraint'], s3_bucket_name, filename)
    data = shortent(object_url)
    print(data)

    return jsonify({'public_url': object_url})


@app.route("/qrcode", methods=['POST'])
def qrcode_post():
    data = request.get_json()
    if data is None:
        print("no json data")
        return "no json data", 400
    if 'qrcode' not in data:
        print("qrcode not in json data")
        return "qrcode not in json data", 400
    if 'content' not in data['qrcode']:
        print("qrcode.content not in json data")
        return "qrcode.content not in json data", 400

    qrcode_params = {
        'version': 10,
        'border': 2,
        'box_size': 10,
        'fill_color': 'black',
        'back_color': 'white',
        'fit': True,
        'content': '',
    }

    if 'version' in data['qrcode']:
        qrcode_params['version'] = int(data['qrcode']['version'])
        if qrcode_params['version'] < 1:
            qrcode_params['version'] = None
    if 'border' in data['qrcode']:
        qrcode_params['border'] = int(data['qrcode']['border'])
    if 'box_size' in data['qrcode']:
        qrcode_params['box_size'] = int(data['qrcode']['box_size'])
    if 'fill_color' in data['qrcode']:
        qrcode_params['fill_color'] = data['qrcode']['fill_color']
    if 'back_color' in data['qrcode']:
        qrcode_params['back_color'] = data['qrcode']['back_color']
    if 'content' in data['qrcode']:
        qrcode_params['content'] = data['qrcode']['content']
    if 'fit' in data['qrcode']:
        qrcode_params['fit'] = data['qrcode'][
            'fit'] in [True, '1', 'true', 'True']

    qr = qrcode.QRCode(
        version=qrcode_params['version'],
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=qrcode_params['box_size'],
        border=qrcode_params['border'], )
    qr.add_data(qrcode_params['content'])
    qr.make(fit=qrcode_params['fit'])
    img = qr.make_image(
        fill_color=qrcode_params['fill_color'],
        back_color=qrcode_params['back_color'])
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')
    img_io.seek(0)

    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='qrcode.png'), 200


@app.route("/twitter", methods=['POST'])
def post_to_twitter():
    data = request.get_json()
    user_handle = data.get('user_handle', 'a NeurIPS attendee')
    link_asset = data.get('link_asset', None)
    if link_asset is None:
        abort(404)
    response = twitter_api.PostUpdate(
        "New creation by {}! @element_ai".format(user_handle),
        media=link_asset)
    return jsonify({'response': 'Posted: {}'.format(link_asset)}), 200
