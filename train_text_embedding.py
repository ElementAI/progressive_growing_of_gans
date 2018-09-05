#!/usr/bin/env python3

"""Training and evaluation entry point."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import PIL
import os
import pickle
import h5py
import glob
import numpy as np
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes
from common.util import ACTIVATION_MAP
from tqdm import trange
import pathlib
import logging
from common.util import summary_writer
from common.gen_experiments import load_and_save_params
import time
from tqdm import tqdm, trange
from keras.preprocessing.sequence import pad_sequences
from dataset_tool import get_json_from_ssense_img_name
from tensorflow.contrib.slim.nets import inception
from shutil import copy
from typing import List, Dict

# TODO: connect to borgy

'''
Execute as a script:
go to deep-prior root and run
export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=0
go to deep-prior/deep_prior/experiements/mini-imagenet and run
python train_text_embedding.py
'''

# docker pull gcr.io/tensorflow/tensorflow:1.4.1-devel-gpu-py3
# docker tag gcr.io/tensorflow/tensorflow:1.4.1-devel-gpu-py3 images.borgy.elementai.lan/tensorflow/tensorflow:1.4.1-devel-gpu-py3
# docker push images.borgy.elementai.lan/tensorflow/tensorflow:1.4.1-devel-gpu-py3

tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)

# --data_dir=/home/boris/Downloads/cifar-100-python
# --data_dir=../../../data/mini-imagenet
DATA_DIR = "/mnt/scratch/ssense/data_dumps/images_png_dump_256"


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'test', 'build_tokenizer'])
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to the data.')
    parser.add_argument('--data_split', type=str, default='sources', choices=['train', 'val', 'test'],
                        help='Split of the data to be used to perform operation.')

    # Training parameters
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--number_of_steps', type=int, default=int(100000),
                        help="Number of training steps (number of Epochs in Hugo's paper)")
    parser.add_argument('--number_of_steps_to_early_stop', type=int, default=int(1000000),
                        help="Number of training steps after half way to early stop the training")
    parser.add_argument('--log_dir', type=str, default='', help='Base log dir')
    parser.add_argument('--exp_dir', type=str, default=None, help='experiement directory for Borgy')
    parser.add_argument('--num_classes_train', type=int, default=5,
                        help='Number of classes in the train phase, this is coming from the prototypical networks')
    parser.add_argument('--num_shots_train', type=int, default=5,
                        help='Number of shots in a few shot meta-train scenario')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--num_tasks_per_batch', type=int, default=1,
                        help='Number of few shot tasks per batch, so the task encoding batch is num_tasks_per_batch x num_classes_test x num_shots_train .')
    parser.add_argument('--init_learning_rate', type=float, default=0.00102, help='Initial learning rate.')
    parser.add_argument('--save_summaries_secs', type=int, default=60, help='Time between saving summaries')
    parser.add_argument('--save_interval_secs', type=int, default=60, help='Time between saving model?')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--augment', type=bool, default=False)
    # Learning rate paramteres
    parser.add_argument('--lr_anneal', type=str, default='exp', choices=['exp'])
    parser.add_argument('--n_lr_decay', type=int, default=3)
    parser.add_argument('--lr_decay_rate', type=float, default=2.0)
    parser.add_argument('--num_steps_decay_pwc', type=int, default=2500,
                        help='Decay learning rate every num_steps_decay_pwc')

    parser.add_argument('--clip_gradient_norm', type=float, default=1.0, help='gradient clip norm.')
    parser.add_argument('--weights_initializer_factor', type=float, default=0.1,
                        help='multiplier in the variance of the initialization noise.')
    # Evaluation parameters
    parser.add_argument('--max_number_of_evaluations', type=float, default=float('inf'))
    parser.add_argument('--eval_interval_secs', type=int, default=120, help='Time between evaluating model?')
    parser.add_argument('--eval_interval_steps', type=int, default=1000,
                        help='Number of train steps between evaluating model in the training loop')
    parser.add_argument('--eval_interval_fine_steps', type=int, default=1000,
                        help='Number of train steps between evaluating model in the training loop in the final phase')
    parser.add_argument('--num_samples_eval', type=int, default=100, help='Number of evaluation samples?')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size?')
    # Test parameters
    parser.add_argument('--num_classes_test', type=int, default=5, help='Number of classes in the test phase')
    parser.add_argument('--num_shots_test', type=int, default=5,
                        help='Number of shots in a few shot meta-test scenario')
    parser.add_argument('--num_cases_test', type=int, default=50000,
                        help='Number of few-shot cases to compute test accuracy')
    parser.add_argument('--pretrained_model_dir', type=str,
                        default='./logs/batch_size-32-num_tasks_per_batch-1-lr-0.122-lr_anneal-cos-epochs-100.0-dropout-1.0-optimizer-sgd-weight_decay-0.0005-augment-False-num_filters-64-feature_extractor-simple_res_net-task_encoder-class_mean-attention_num_filters-32/train',
                        help='Path to the pretrained model to run the nearest neigbor baseline test.')
    # Architecture parameters
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    # Image feature extractor
    parser.add_argument('--image_feature_extractor', type=str, default='simple_res_net',
                        choices=['simple_res_net', 'inception_v3'], help='Which feature extractor to use')
    parser.add_argument('--image_fe_trainable', type=bool, default=True)
    parser.add_argument('--image_fe_checkpoint_file', type=str, default='/mnt/scratch/ssense/data_dumps/pretrained_models/inception_v3.ckpt') # '/Users/boris/Downloads/inception_v3.ckpt'
    parser.add_argument('--num_filters', type=int, default=64)
    parser.add_argument('--num_units_in_block', type=int, default=3)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--num_max_pools', type=int, default=3)
    parser.add_argument('--block_size_growth', type=float, default=2.0)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'selu', 'swish-1'])
    # Text feature extractor
    parser.add_argument('--word_embed_dim', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--text_feature_extractor', type=str, default='simple_bi_lstm', choices=['simple_bi_lstm'])

    parser.add_argument('--embedding_size', type=int, default=None)
    parser.add_argument('--embedding_pooled', type=bool, default=True)

    parser.add_argument('--metric_multiplier_init', type=float, default=10.0, help='multiplier of cosine metric')
    parser.add_argument('--metric_multiplier_trainable', type=bool, default=False,
                        help='multiplier of cosine metric trainability')
    parser.add_argument('--polynomial_metric_order', type=int, default=1)


    args = parser.parse_args()

    print(args)
    return args


def get_image_size(data_dir : str):
    """ Generates image size based on the dataset directory name

    :param data_dir: path to the data
    :return: image size
    """

    return 256


class Namespace(object):
    """
    Wrapper around dictionary to make it saveable
    """
    def __init__(self, adict):
        self.__dict__.update(adict)


def get_logdir_name(flags):
    """Generates the name of the log directory from the values of flags
    Parameters
    ----------
        flags: neural net architecture generated by get_arguments()
    Outputs
    -------
        the name of the directory to store the training and evaluation results
    """

    param_list = ['batch_size', str(flags.train_batch_size), 'steps', str(flags.number_of_steps),
                  'lr', str(flags.init_learning_rate), 'opt', flags.optimizer,
                  'weight_decay', str(flags.weight_decay),
                  'nfilt', str(flags.num_filters), 'image_feature_extractor', str(flags.image_feature_extractor),
                  ]

    if flags.log_dir == '':
        logdir = './logs/' + '-'.join(param_list)
    else:
        logdir = os.path.join(flags.log_dir, '-'.join(param_list))

    if flags.exp_dir is not None:
        # Running a Borgy experiment
        logdir = flags.exp_dir

    return logdir


class ScaledVarianceRandomNormal(init_ops.Initializer):
    """Initializer that generates tensors with a normal distribution scaled as per https://arxiv.org/pdf/1502.01852.pdf.
    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      dtype: The data type. Only floating point types are supported.
    """

    def __init__(self, mean=0.0, factor=1.0, seed=None, dtype=dtypes.float32):
        self.mean = mean
        self.factor = factor
        self.seed = seed
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        if shape:
            n = float(shape[-1])
        else:
            n = 1.0
        for dim in shape[:-2]:
            n *= float(dim)

        self.stddev = np.sqrt(self.factor * 2.0 / n)
        return random_ops.random_normal(shape, self.mean, self.stddev,
                                        dtype, seed=self.seed)


def _get_scope(is_training, flags):
    """
    Get slim scope parameters for the convolutional and dropout layers

    :param is_training: whether the network is in training mode
    :param flags: overall settings of the model
    :return: convolutional and dropout scopes
    """
    normalizer_params = {
        'epsilon': 1e-6,
        'decay': .95,
        'center': True,
        'scale': True,
        'trainable': True,
        'is_training': is_training,
        'updates_collections': None, # tf.GraphKeys.UPDATE_OPS
        'param_regularizers': {'beta': tf.contrib.layers.l2_regularizer(scale=flags.weight_decay),
                               'gamma': tf.contrib.layers.l2_regularizer(scale=flags.weight_decay)},
    }
    conv2d_arg_scope = slim.arg_scope(
        [slim.conv2d],
        activation_fn=ACTIVATION_MAP[flags.activation],
        normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params=normalizer_params,
        # padding='SAME',
        trainable=is_training,
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.weight_decay),
        weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor),
        biases_initializer=tf.constant_initializer(0.0)
    )
    dropout_arg_scope = slim.arg_scope(
        [slim.dropout],
        keep_prob=flags.dropout,
        is_training=is_training)
    return conv2d_arg_scope, dropout_arg_scope


def get_simple_res_net(images, flags, num_filters, is_training=False, reuse=None, scope=None):

    conv2d_arg_scope, dropout_arg_scope = _get_scope(is_training, flags)
    activation_fn = ACTIVATION_MAP[flags.activation]
    with conv2d_arg_scope, dropout_arg_scope:
        with tf.variable_scope(scope or 'image_feature_extractor', reuse=reuse):
            # h = slim.conv2d(images, num_outputs=num_filters[0], kernel_size=6, stride=1,
            #                 scope='conv_input', padding='SAME')
            # h = slim.max_pool2d(h, kernel_size=2, stride=2, padding='SAME', scope='max_pool_input')
            h = images
            for i in range(len(num_filters)):
                # make shortcut
                shortcut = slim.conv2d(h, num_outputs=num_filters[i], kernel_size=1, stride=1,
                                       activation_fn=None,
                                       scope='shortcut' + str(i), padding='SAME')

                for j in range(flags.num_units_in_block):
                    h = slim.conv2d(h, num_outputs=num_filters[i], kernel_size=3, stride=1,
                                    scope='conv' + str(i) + '_' + str(j), padding='SAME', activation_fn=None)

                    if j < (flags.num_units_in_block - 1):
                        h = activation_fn(h, name='activation_' + str(i) + '_' + str(j))
                h = h + shortcut

                h = activation_fn(h, name='activation_' + str(i) + '_' + str(flags.num_units_in_block - 1))
                if i < (len(num_filters)-1):
                    h = slim.max_pool2d(h, kernel_size=2, stride=2, padding='SAME', scope='max_pool' + str(i))

            if flags.embedding_pooled:
                kernel_size = h.shape.as_list()[-2]
                h = slim.avg_pool2d(h, kernel_size=kernel_size, scope='avg_pool')
            h = slim.flatten(h)

            if flags.dropout:
                h = slim.dropout(h, scope='fc_dropout', keep_prob=1.0 - flags.dropout)

            # Bottleneck layer
            if flags.embedding_size:
                h = slim.fully_connected(h, num_outputs=flags.embedding_size,
                                         activation_fn=activation_fn, normalizer_fn=None,
                                         scope='image_feature_adaptor')
    return h


def get_inception_v3(images, flags, is_training=False, reuse=None, scope=None):
    arg_scope = inception.inception_v3_arg_scope()
    training_flag = is_training and flags.image_fe_trainable
    with slim.arg_scope(arg_scope):
        with tf.variable_scope(scope or 'image_feature_extractor', reuse=reuse):
            images = tf.image.resize_bilinear(images, size=[299, 299], align_corners=False)
            scaled_input_tensor = tf.scalar_mul((1.0 / 255), images)
            scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
            scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

            logits, end_points = inception.inception_v3(scaled_input_tensor, is_training=training_flag, num_classes=1001,
                                                        reuse=reuse)
            h = end_points['PreLogits']
            h = slim.flatten(h)
    return h


def get_image_feature_extractor(images, flags, is_training=False, scope='image_feature_extractor', reuse=None):
    """
        Image feature extractor selector
    :param images: tensor of input images in the format BHWC
    :param flags: overall architecture settings
    :param num_filters:
    :param is_training:
    :param scope:
    :param reuse:
    :return:
    """
    num_filters = [round(flags.num_filters * pow(flags.block_size_growth, i)) for i in range(flags.num_blocks)]
    if flags.image_feature_extractor == 'simple_res_net':
        h = get_simple_res_net(images, flags=flags, num_filters=num_filters, is_training=is_training, reuse=reuse, scope=scope)
    elif flags.image_feature_extractor == 'inception_v3':
        h = get_inception_v3(images, flags=flags, is_training=is_training, reuse=reuse, scope=scope)
    return h


def get_simple_bi_lstm(text, text_length, flags, embedding_size=512, is_training=False, scope='text_feature_extractor', reuse=None):
    """

    :param text: input text sequence, BTC
    :param text_length:  lengths of sequences in the batch, B
    :param flags:  general settings of the overall architecture
    :param is_training:
    :param scope:
    :param reuse:
    :return: the text embedding, BC
    """

    with tf.variable_scope(scope, reuse=reuse):
        h = tf.contrib.layers.embed_sequence(text,
                                             vocab_size=flags.vocab_size,
                                             embed_dim=flags.word_embed_dim,
                                             trainable=is_training,
                                             scope='TextEmbedding')

        cells_fw = [tf.nn.rnn_cell.LSTMCell(size) for size in [256]]
        cells_bw = [tf.nn.rnn_cell.LSTMCell(size) for size in [256]]
        initial_states_fw = [cell.zero_state(text.get_shape()[0], dtype=tf.float32) for cell in cells_fw]
        initial_states_bw = [cell.zero_state(text.get_shape()[0], dtype=tf.float32) for cell in cells_bw]

        h, *_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                               cells_bw=cells_bw,
                                                               inputs=h,
                                                               initial_states_fw=initial_states_fw,
                                                               initial_states_bw=initial_states_bw,
                                                               dtype=tf.float32,
                                                               sequence_length=text_length)
        mask = tf.expand_dims(tf.sequence_mask(text_length, maxlen=tf.shape(text)[1], dtype=tf.float32), axis=-1)
        h = tf.reduce_sum(tf.multiply(h, mask), axis=[1]) / tf.reduce_sum(mask, axis=[1])
        # this is the adaptor to match the size of the image extractor
#         h = tf.contrib.layers.fully_connected(h, num_outputs=embedding_size)
    return h


def get_text_feature_extractor(text, text_length, flags, embedding_size=512, is_training=False, scope='text_feature_extractor', reuse=None):
    """
        Text extractor selector
    :param text: tensor of input texts tokenized as integers in the format BL
    :param text_length: tensor of sequence lengths
    :param flags: overall architecture settings
    :param embedding_size: the length of embedding vector
    :param is_training:
    :param scope:
    :param reuse:
    :return:
    """
    if flags.text_feature_extractor == 'simple_bi_lstm':
        h = get_simple_bi_lstm(text, text_length, flags=flags, embedding_size=embedding_size, is_training=is_training,
                               reuse=reuse, scope=scope)
    return h


def get_distance_head(embedding_mod1, embedding_mod2, flags, is_training, scope='distance_head'):
    """
        Implements the a distance head, measuring distance between elements in embedding_mod1 and embedding_mod2.
        Input dimensions are B1C and B2C, output dimentions are B1B2. The distance between diagonal elements is supposed to be small.
        The distance between off-diagonal elements is supposed to be large. Output can be considered to be classification logits.
    :param embedding_mod1: embedding of modality one, B1C
    :param embedding_mod2: embeddings of modality two, B2C
    :param flags: general architecture parameters
    :param is_training:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        # i is the number of elements in the embedding_mod1 batch
        # j is the number of elements in the embedding_mod2 batch
        j = embedding_mod2.get_shape()[0]
        i = embedding_mod1.get_shape()[0]
        # tile to be able to produce weight matrix alpha in (i,j) space
        embedding_mod1 = tf.expand_dims(embedding_mod1, axis=1)
        embedding_mod2 = tf.expand_dims(embedding_mod2, axis=0)
        # features_generic changes over i and is constant over j
        # task_encoding changes over j and is constant over i
        embedding_mod2_tile = tf.tile(embedding_mod2, (i, 1, 1))
        embedding_mod1_tile = tf.tile(embedding_mod1, (1, j, 1))
        # Compute distance
        euclidian = -tf.norm(embedding_mod2_tile-embedding_mod1_tile, name='neg_euclidian_distance', axis=-1)
        return euclidian


def get_inference_graph(images, text, text_length, flags, is_training):
    """
        Creates text embedding, image embedding and links them using a distance metric.
        Ouputs logits that can be used for training and inference, as well as text and image embeddings.
    :param images:
    :param text:
    :param labels:
    :param flags:
    :param is_training:
    :return:
    """

    with tf.variable_scope('Model'):
        image_embeddings = get_image_feature_extractor(images, flags, is_training=is_training,
                                                       scope='image_feature_extractor', reuse=False)
        text_embedding_size = image_embeddings.get_shape().as_list()[-1]
        text_embeddings = get_text_feature_extractor(text, text_length, flags, embedding_size=text_embedding_size,
                                                     is_training=is_training, scope='text_feature_extractor', reuse=False)

        # Here we compute logits of correctly matching text to a given image.
        # We could also compute logits of correctly matching an image to a given text by reversing
        # image_embeddings and text_embeddings
        logits = get_distance_head(embedding_mod1=image_embeddings,
                                   embedding_mod2=text_embeddings, flags=flags,
                                   is_training=is_training, scope='distance_head')
    return logits, image_embeddings, text_embeddings


def get_input_placeholders(batch_size, image_size, scope):
    """
    :param batch_size:
    :return: placeholders for images, text and class labels
    """
    with tf.variable_scope(scope):
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3), name='images')
        text_placeholder = tf.placeholder(shape=(batch_size, None), name='text', dtype=tf.int32)
        text_length_placeholder = tf.placeholder(shape=(batch_size,), name='text_len', dtype=tf.int32)
        labels_txt2img = tf.placeholder(tf.int64, shape=(batch_size,), name='match_labels_txt2img')
        labels_img2txt = tf.placeholder(tf.int64, shape=(batch_size,), name='match_labels_img2txt')
        return images_placeholder, text_placeholder, text_length_placeholder, labels_txt2img, labels_img2txt


def get_lr(global_step=None, flags=None):
    """
    Creates a learning rate schedule
    :param global_step: external global step variable, if None new one is created here
    :param flags:
    :return:
    """
    if global_step is None:
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

    if flags.lr_anneal == 'exp':
        lr_decay_step = flags.number_of_steps // flags.n_lr_decay
        learning_rate = tf.train.exponential_decay(flags.init_learning_rate, global_step, lr_decay_step,
                                                   1.0 / flags.lr_decay_rate, staircase=True)
    else:
        raise Exception('Learning rate schedule not implemented')

    tf.summary.scalar('learning_rate', learning_rate)
    return learning_rate


def get_main_train_op(loss: tf.Tensor, global_step: tf.Variable, flags: Namespace):
    """
    Creates a train operation to minimize loss
    :param loss: loss to be minimized
    :param global_step: global step to be incremented whilst invoking train opeation created
    :param flags: overall architecture parameters
    :return:
    """

    # Learning rate
    learning_rate = get_lr(global_step=global_step, flags=flags)
    # Optimizer
    if flags.optimizer == 'sgd':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif flags.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        raise Exception('Optimizer not implemented')
    # get variables to train
    if flags.image_fe_trainable:
        variables_to_train = tf.trainable_variables()
    else:
        variables_to_train = tf.trainable_variables(scope='(?!.*image_feature_extractor).*')
    # Train operation
    return slim.learning.create_train_op(total_loss=loss, optimizer=optimizer, global_step=global_step,
                                         clip_gradient_norm=flags.clip_gradient_norm,
                                         variables_to_train=variables_to_train)


class SsenseDataset(object):
    """ Basic image and text dataset generating batches from a collection of files in a folder """

    def __init__(self, data_path="/mnt/scratch/ssense/data_dumps/images_png_dump_256",
                 tokenizer_path='/mnt/scratch/boris/ssense/tokenizer_embedding.pkl',
                 maxlen=100):
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path

        glob_pattern = os.path.join(data_path, '*.png')
        self.image_filenames = sorted(glob.glob(glob_pattern))
        self.n_samples = len(self.image_filenames)
        self.maxlen = maxlen

        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def get_images(self, img_names):
        images = []
        for img_name in img_names:
            images.append(np.asarray(PIL.Image.open(img_name)))
        return np.asarray(images)

    def get_text(self, img_names):
        texts = []
        text_length = []
        for img_name in img_names:
            json_content = get_json_from_ssense_img_name(self.data_path, img_name)
            # TODO: Remove b' at the beginnning and ' at the end, this is dirty solution
            text_current = json_content['description'][2:-1]
            text_tokens = self.tokenizer.texts_to_sequences([text_current])[0]
            text_length.append(len(text_tokens))
            texts.append(text_tokens)
        texts = pad_sequences(texts, maxlen=self.maxlen, padding='post')
        return texts, np.asarray(text_length, dtype=np.int32)

    def next_batch(self, batch_size=64):
        idxs = np.random.randint(self.n_samples, size=batch_size)
        img_names = [self.image_filenames[i] for i in idxs]

        images = self.get_images(img_names)
        texts, text_length = self.get_text(img_names)
        # Shuffling the text to avoid having a trivial relation between image indexes and text indexes
        permutation = np.random.choice(np.arange(batch_size, dtype=np.int32), size=batch_size, replace=False)
        labels_txt2img = np.arange(batch_size, dtype=np.int32)
        for i in range(batch_size):
            labels_txt2img[permutation[i]] = i
        labels_img2txt = permutation
        return images, texts[permutation], text_length[permutation], (labels_txt2img, labels_img2txt)

    def sequential_batches(self, batch_size, n_batches, rng=np.random):
        """Generator for a random sequence of minibatches with no overlap."""
        permutation = rng.permutation(self.n_samples)
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = np.minimum((start + batch_size), self.n_samples)
            idxs = permutation[start:end]
            img_names = [self.image_filenames[i] for i in idxs]

            images = self.get_images(img_names)
            texts, text_length = self.get_text(img_names)

            yield images, texts, text_length
            if end == self.n_samples:
                break


def get_train_datasets(data_path="/mnt/scratch/ssense/data_dumps/images_png_dump_256",
                       tokenizer_path='/mnt/scratch/boris/ssense/tokenizer_embedding.pkl',
                       maxlen=100,
                       splits: List[str]=['train', 'test', 'validation']):
    
    assert set(splits).issubset(next(os.walk(data_path))[1]), "One or more splits doesn't exist in %s" %data_path
    datasets = {}
    for split_folder in splits:
        datasets[split_folder] = SsenseDataset(data_path=os.path.join(data_path, split_folder),
                                              tokenizer_path=tokenizer_path, maxlen=maxlen)
    return datasets


class ModelLoader:
    def __init__(self, model_path, batch_size):
        self.batch_size = batch_size

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=os.path.join(model_path, 'train'))
        step = int(os.path.basename(latest_checkpoint).split('-')[1])

        flags = Namespace(load_and_save_params(default_params=dict(), exp_dir=model_path))
        image_size = get_image_size(flags.data_dir)

        with tf.Graph().as_default():
            images_pl, text_pl, text_len_pl, match_labels_txt2img, match_labels_img2txt = get_input_placeholders(
                batch_size=batch_size,
                image_size=image_size, scope='inputs')
            logits, image_embeddings, text_embeddings = get_inference_graph(
                images=images_pl, text=text_pl, text_length=text_len_pl, flags=flags, is_training=False)
            self.images_pl = images_pl
            self.text_pl = text_pl
            self.text_len_pl = text_len_pl
            self.match_labels_txt2img_pl = match_labels_txt2img
            self.match_labels_img2txt_pl = match_labels_img2txt
            self.image_embeddings = image_embeddings
            self.text_embeddings = text_embeddings

            init_fn = slim.assign_from_checkpoint_fn(latest_checkpoint, tf.global_variables())

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # Run init before loading the weights
            self.sess.run(tf.global_variables_initializer())
            # Load weights
            init_fn(self.sess)

            self.flags = flags
            self.logits = logits
            self.logits_size = self.logits.get_shape().as_list()[-1]
            self.step = step

    def predict(self, images, texts, text_len):
        feed_dict = {self.images_pl: images.astype(dtype=np.float32),
                     self.text_pl: texts,
                     self.text_len_pl: text_len}
        return self.sess.run([self.logits, self.image_embeddings, self.text_embeddings], feed_dict)

    def eval(self, data_set: Dict[str, SsenseDataset]=None, num_samples: int=100):
        """
        Runs evaluation loop over dataset
        :param data_set:
        :param num_samples: number of tasks to sample from the dataset
        :return:
        """
        num_correct_txt2img = 0.0
        num_correct_img2txt = 0.0
        num_tot = 0.0
        for i in trange(num_samples):
            images, texts, text_len, match_labels = data_set.next_batch(batch_size=self.batch_size)
            labels_txt2img, labels_img2txt = match_labels
            feed_dict = {self.images_pl: images.astype(dtype=np.float32),
                         self.text_pl: texts,
                         self.text_len_pl: text_len}
            logits = self.sess.run(self.logits, feed_dict)
            labels_pred_txt2img = np.argmax(logits, axis=-1)
            labels_pred_img2txt = np.argmax(logits, axis=0)

            num_correct_txt2img += sum(labels_pred_txt2img == labels_txt2img)
            num_correct_img2txt += sum(labels_pred_img2txt == labels_img2txt)
            num_tot += len(labels_pred_txt2img)
        return {'acc_txt2img': num_correct_txt2img / num_tot, 'acc_img2txt': num_correct_img2txt / num_tot}


def eval_once(flags: Namespace, datasets: Dict[str, SsenseDataset]):
    
    model = ModelLoader(model_path=flags.pretrained_model_dir, batch_size=flags.eval_batch_size)
    results = {}
    for data_name, dataset in datasets.items():
        results_eval = model.eval(data_set=dataset, num_samples=flags.num_samples_eval)
        for result_name, result_val in results_eval.items():
            results["evaluation/"+result_name+"_"+data_name] = result_val
            logging.info("accuracy_%s: %.3g" % (result_name+"_"+data_name, result_val))
    
    log_dir = get_logdir_name(flags)
    eval_writer = summary_writer(log_dir + '/eval')
    eval_writer(model.step, **results)
    

def get_image_fe_restorer(flags : Namespace):
    if flags.image_feature_extractor == 'inception_v3':
        vars = tf.get_collection(key=tf.GraphKeys.MODEL_VARIABLES, scope='.*InceptionV3')

        def name_in_checkpoint(var: tf.Variable):
            return '/'.join(var.op.name.split('/')[2:])

        return tf.train.Saver(var_list={name_in_checkpoint(var): var for var in vars})
    else:
        return None


def test_pretrained_inception_model(images_pl, sess):
    # code to test loaded inception model
    sample_images = ['dog.jpg', 'panda.jpg', 'tinca_tinca.jpg']
    from PIL import Image
    graph = tf.get_default_graph()
    inception_logits_pl = graph.get_tensor_by_name("Model/image_feature_extractor/InceptionV3/Predictions/Reshape_1:0")
    for image in sample_images:
        im = Image.open(image).resize((256,256))
        im = np.array(im)
        im = im.reshape(-1,256,256,3).astype(np.float32)
        im = np.tile(im, [images_pl.get_shape().as_list()[0], 1, 1, 1])
        logit_values = sess.run(inception_logits_pl, feed_dict={images_pl: im})
        print(image)
        print(np.max(logit_values, axis=-1))
        print(np.argmax(logit_values, axis=-1)-1)
    
        
def train(flags):
    log_dir = get_logdir_name(flags)
    flags.pretrained_model_dir = log_dir
    log_dir = os.path.join(log_dir, 'train')
    # This is setting to run evaluation loop only once
    flags.max_number_of_evaluations = 1
    flags.eval_interval_secs = 0
    image_size = get_image_size(flags.data_dir)

    # Get datasets
    datasets = get_train_datasets()
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        images_pl, text_pl, text_len_pl, match_labels_txt2img_pl, match_labels_img2txt_pl = \
            get_input_placeholders(batch_size=flags.train_batch_size,
                                   image_size=image_size, scope='inputs')

        logits, image_embeddings, text_embeddings = get_inference_graph(images=images_pl, text=text_pl,
                                                                        text_length=text_len_pl, flags=flags,
                                                                        is_training=True)
        loss_txt2img = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=tf.one_hot(match_labels_txt2img_pl, flags.train_batch_size)),
                                                    name='loss_txt2img')
        loss_img2txt = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(logits, perm=[1, 0]),
                                                    labels=tf.one_hot(match_labels_img2txt_pl, flags.train_batch_size)),
                                                    name='loss_img2txt')

        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_tot = tf.add_n([0.5*loss_txt2img + 0.5*loss_img2txt] + regu_losses)
        misclass_txt2img = 1.0 - slim.metrics.accuracy(tf.argmax(logits, 1), match_labels_txt2img_pl)
        misclass_img2txt = 1.0 - slim.metrics.accuracy(tf.argmax(logits, 0), match_labels_img2txt_pl)
        main_train_op = get_main_train_op(loss_tot, global_step, flags)

        tf.summary.scalar('loss/total', loss_tot)
        tf.summary.scalar('loss/txt2img', loss_txt2img)
        tf.summary.scalar('loss/img2txt', loss_img2txt)
        tf.summary.scalar('misclassification/txt2img', misclass_txt2img)
        tf.summary.scalar('misclassification/img2txt', misclass_img2txt)
        summary = tf.summary.merge(tf.get_collection('summaries'))

        # Define session and logging
        summary_writer = tf.summary.FileWriter(log_dir, flush_secs=1)
        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        image_fe_restorer = get_image_fe_restorer(flags=flags)
        supervisor = tf.train.Supervisor(logdir=log_dir, init_feed_dict=None,
                                         summary_op=None,
                                         init_op=tf.global_variables_initializer(),
                                         summary_writer=summary_writer,
                                         saver=saver,
                                         global_step=global_step, save_summaries_secs=flags.save_summaries_secs,
                                         save_model_secs=0)

        with supervisor.managed_session() as sess:
            if image_fe_restorer:
                image_fe_restorer.restore(sess, flags.image_fe_checkpoint_file)
                # test_pretrained_inception_model(images_pl, sess)
            
            checkpoint_step = sess.run(global_step)
            if checkpoint_step > 0:
                checkpoint_step += 1

            loss_tot, dt_train = 0.0, 0.0
            for step in range(checkpoint_step, flags.number_of_steps):
                # get batch of data to compute classification loss
                dt_batch = time.time()
                images, text, text_length, match_labels = datasets['train'].next_batch(batch_size=flags.train_batch_size)
                labels_txt2img, labels_img2txt = match_labels
                dt_batch = time.time() - dt_batch
                # if flags.augment:
                #     images = image_augment(images)
                
                feed_dict = {images_pl: images.astype(dtype=np.float32), text_len_pl: text_length,
                             text_pl: text,
                             match_labels_txt2img_pl: labels_txt2img, match_labels_img2txt_pl: labels_img2txt}

                if step % 100 == 0:
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    logging.info("step %d, loss : %.4g, dt: %.3gs, dt_batch: %.3gs" % (step, loss_tot, dt_train, dt_batch))
                    
                if step % 100 == 0:
                    logits_img2txt = sess.run(logits, feed_dict=feed_dict)
                    logits_img2txt = np.argmax(logits_img2txt, axis=0)
                    num_matches = float(sum(labels_img2txt == logits_img2txt))
                    logging.info("img2txt acc: %.3g" %(num_matches/flags.train_batch_size))

                t_train = time.time()
                loss_tot = sess.run(main_train_op, feed_dict=feed_dict)
                dt_train = time.time() - t_train

                if step % flags.eval_interval_steps == 0:
                    saver.save(sess, os.path.join(log_dir, 'model'), global_step=step)
                    eval_once(flags, datasets=datasets)


def test():
    return None


def build_tokenizer(flags):

    from keras.preprocessing.text import Tokenizer
    if os.path.exists('/mnt/scratch/boris/ssense/tokenizer_embedding.pkl'):
        with open('/mnt/scratch/boris/ssense/tokenizer_embedding.pkl', 'rb') as input_file:
            tokenizer = pickle.load(input_file)
    else:
        tokenizer = Tokenizer(num_words=flags.vocab_size, filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

        f = h5py.File('/mnt/scratch/ssense/latest_indexes/unlocked_indexes/ssense_256_256_train.h5', mode='r')
        descriptions = np.char.decode(f['input_description'][:, 0], 'latin')
        tokenizer.fit_on_texts(list(descriptions))
        f.close()
        with open('/mnt/scratch/boris/ssense/tokenizer_embedding.pkl', 'wb') as output_file:
            pickle.dump(tokenizer, output_file)


def get_product_id(img_name : str):
    file_name, ext = os.path.splitext(os.path.basename(img_name))
    product_id, *_, pose_id = file_name.split('_')
    return product_id


def get_product_list(image_filenames : list):
    return list(set([get_product_id(img_name) for img_name in image_filenames]))


def get_product_to_filename_map(image_filenames : list):
    prod_to_filename_map = {}
    for img_name in image_filenames:
        prod_id = get_product_id(img_name)
        prod_to_filename_map[prod_id] = prod_to_filename_map.get(prod_id, []) + [img_name]
    return prod_to_filename_map


def get_splits(image_filenames: List[str], test_split: Dict[str, float]) -> Dict[str, List[str]]:
    
    if test_split:
        product_ids = get_product_list(image_filenames)
        print("Loaded %d product ids..." %(len(product_ids)))
        
        print("Create train/test splits in the product id space")
        split_prod_id_dict = {}
        split_idxs = {}
        product_idxs = np.arange(len(product_ids), dtype=np.int64)
        assert sum(test_split.values()) == 1.0, "Split probabilities do not sum up to 1"
        for split_name, split_frac in test_split.items():
            size = int(round(split_frac * len(product_ids)))
            split_idxs[split_name] = np.random.choice(product_idxs, size=size, replace=False)
            split_prod_id_dict[split_name] = [product_ids[idx] for idx in split_idxs[split_name]]
            product_idxs = np.setxor1d(product_idxs, split_idxs[split_name])
                
        print("Test if product ID splits contain all product IDs ...")
        assert set(np.concatenate(list(split_idxs.values()))) == set(np.arange(len(product_ids), dtype=np.int64))
        print("Test if product ID splits are disjoint...")
        for split_name1, split_idxs1 in split_idxs.items():
            for split_name2, split_idxs2 in split_idxs.items():
                if split_name1 != split_name2:
                    assert len(set.intersection(set(split_idxs1), set(split_idxs2))) == 0, "Splits overlap"
        
        print("Create train/test splits in the image filename space")
        split_dict = {}
        prod_to_filename_map = get_product_to_filename_map(image_filenames)
        for split_name, split_frac in test_split.items():
            split_product_ids = split_prod_id_dict[split_name]
            split_filenames = [prod_to_filename_map[prod_id] for prod_id in split_product_ids]
            # Flatten list of lists
            split_filenames = [item for sublist in split_filenames for item in sublist]
            split_dict[split_name] = split_filenames
        
        print("Test if filename splits contain all filenames ...")
        assert set(np.concatenate(list(split_dict.values()))) == set(image_filenames)
        print("Test if filename splits are disjoint...")
        for split_name1, split_idxs1 in split_dict.items():
            for split_name2, split_idxs2 in split_dict.items():
                if split_name1 != split_name2:
                    assert len(set.intersection(set(split_idxs1), set(split_idxs2))) == 0, "Splits overlap"
                    
        for split_name, split_filenames in split_dict.items():
            print("Number of files in split '%s': %d" %(split_name, len(split_filenames)))
    else:
        split_dict = {'': image_filenames}
        
    return split_dict


def write_splis_to_disk(split_dict: Dict[str, List[str]], ssense_dir: str, ssense_dir_resized: str, resolution: int=256):
    input_metadata_dir = os.path.join(ssense_dir, 'images_metadata')
    for split_name, split_image_filenames in split_dict.items():
        output_dir = os.path.join(ssense_dir_resized, split_name)
        output_metadata_dir = os.path.join(output_dir, 'images_metadata')
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_metadata_dir).mkdir(parents=True, exist_ok=True)
        for img_name in tqdm(split_image_filenames):
            img = PIL.Image.open(os.path.join(img_name))
            img_size, _ = img.size
            if resolution != img_size:
                img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
            _, tail = os.path.split(img_name)
            img_name_out = os.path.join(output_dir, tail)
            img.save(img_name_out)
            img.close()
            
            prod_id = get_product_id(img_name)
            copy(os.path.join(input_metadata_dir, prod_id+'.json'), output_metadata_dir)

            
def create_png_dump_resized(ssense_dir, ssense_dir_resized, resolution:int=256, test_split=None):
    glob_pattern = os.path.join(ssense_dir, '*.png')
    print("Loading data from: ", ssense_dir)
    image_filenames = sorted(glob.glob(glob_pattern))
    
    split_dict = get_splits(image_filenames=image_filenames, test_split=test_split)
    
    write_splis_to_disk(split_dict=split_dict, ssense_dir=ssense_dir, 
                        ssense_dir_resized=ssense_dir_resized, resolution=resolution)


def main(argv=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print(os.getcwd())

    default_params = get_arguments()
    log_dir =get_logdir_name(flags=default_params)

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    # This makes sure that we can store a json and recove a namespace back
    flags = Namespace(load_and_save_params(vars(default_params), log_dir))

    if flags.mode == 'train':
        train(flags=flags)
    elif flags.mode == 'eval':
        eval(flags=flags, is_primary=True)
    elif flags.mode == 'test':
        test(flags=flags)
    elif flags.mode == 'build_tokenizer':
        build_tokenizer()

if __name__ == '__main__':
    tf.app.run()