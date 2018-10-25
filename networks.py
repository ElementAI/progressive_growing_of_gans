# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf
import tfutil

# NOTE: Do not import any application-specific modules here!

#----------------------------------------------------------------------------


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_clip(a, b, t):
    return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


def cset(cur_lambda, new_cond, new_lambda):
    return lambda: tf.cond(new_cond, new_lambda, cur_lambda)


#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.


def get_weight(shape,
               gain=np.sqrt(2),
               use_wscale=False,
               fan_in=None,
               name='weight'):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable(
            name, shape=shape,
            initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.initializers.random_normal(0, std))


#----------------------------------------------------------------------------
# Fully-connected layer.


def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False, epsilon=1e-8):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = spectral_norm(w, epsilon=epsilon)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)


#----------------------------------------------------------------------------
# Convolutional layer.


def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False, epsilon=1e-8):
    assert kernel >= 1 and kernel % 2 == 1
    print("in conv2d")
    print(x)
    w = get_weight(
        [kernel, kernel, x.shape[1].value, fmaps],
        gain=gain,
        use_wscale=use_wscale)
    w = spectral_norm(w, epsilon=epsilon)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(
        x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')


#----------------------------------------------------------------------------
# FILM Layer postmultipliers


def get_film_postmultiplier(weight_decay_film):
    beta_postmultiplier = tf.get_variable(
        name='beta_postmultiplier',
        dtype=tf.float32,
        initializer=0.0,
        regularizer=tf.contrib.layers.l2_regularizer(
            scale=weight_decay_film, scope='penalize_beta'))
    gamma_postmultiplier = tf.get_variable(
        name='gamma_postmultiplier',
        dtype=tf.float32,
        initializer=0.0,
        regularizer=tf.contrib.layers.l2_regularizer(
            scale=weight_decay_film, scope='penalize_gamma'))
    return beta_postmultiplier, gamma_postmultiplier


#----------------------------------------------------------------------------
# Batch Instance-wise feature vector normalization.


def instance_norm(x, epsilon=1e-8):
    with tf.variable_scope('InstanceNorm'):
        mu = tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        rsigma = tf.rsqrt(
            tf.reduce_mean(tf.square(x), axis=[2, 3], keepdims=True) + epsilon)
        return (x - mu) * rsigma


#----------------------------------------------------------------------------
# FILM Layer


def apply_film(x, text_embed, weight_decay_film, **kwargs):
    embed_size = text_embed.shape.as_list()[1]
    fmaps = x.shape.as_list()[1]
    scope = "".join(tf.get_variable_scope().name.split('/')[-2:])
    with tf.variable_scope('FILM'):
        gamma_weight = get_weight(
            shape=[embed_size, fmaps], use_wscale=False, name='gamma_weight')
        beta_weight = get_weight(
            shape=[embed_size, fmaps], use_wscale=False, name='beta_weight')

        beta_postmultiplier, gamma_postmultiplier = get_film_postmultiplier(
            weight_decay_film)
        gamma_weight = tf.cast(gamma_weight, x.dtype)
        beta_weight = tf.cast(beta_weight, x.dtype)
        beta_postmultiplier = tf.cast(beta_postmultiplier, x.dtype)
        gamma_postmultiplier = tf.cast(gamma_postmultiplier, x.dtype)
        beta_postmultiplier = tfutil.autosummary('Film/beta_0/' + scope,
                                                 beta_postmultiplier)
        gamma_postmultiplier = tfutil.autosummary('Film/gamma_0/' + scope,
                                                  gamma_postmultiplier)

        gamma = tf.matmul(text_embed, gamma_weight)
        beta = tf.matmul(text_embed, beta_weight)

        beta = tf.multiply(beta_postmultiplier, beta, name='postmultiply_beta')
        gamma = 1.0 + tf.multiply(
            gamma_postmultiplier, gamma, name='postmultiply_gamma')

        x = instance_norm(x)

        if len(x.shape) == 2:
            return x * gamma + beta
        else:
            return x * tf.reshape(gamma, [-1, fmaps, 1, 1]) + tf.reshape(
                beta, [-1, fmaps, 1, 1])


#----------------------------------------------------------------------------
# Function to transform condition, e.g. labels or text embedding before feeding into film layers


def embed_condition(x, fmaps, epsilon=1e-8):

    with tf.variable_scope('TextEmbedding'):
        x = dense(x, 2 * fmaps, epsilon=epsilon)
        x = apply_bias(x)
        x = leaky_relu(x)
        mu = x[:, :fmaps]
        log_sigma = x[:, fmaps:]

        embedding_kl_loss = KL_loss(mu, log_sigma)

        eps = tf.truncated_normal(tf.shape(mu))
        stddev = tf.exp(log_sigma)
        text_embed = mu + stddev * eps

    return text_embed, embedding_kl_loss


#----------------------------------------------------------------------------
# reduce_mean normalize also the dimension of the embeddings
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss


#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.


def apply_bias(x):
    b = tf.get_variable(
        'bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])


#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.


def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)


#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x


#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.


def upscale2d_conv2d(x,
                     fmaps,
                     kernel,
                     gain=np.sqrt(2),
                     use_wscale=False,
                     epsilon=1e-8):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight(
        [kernel, kernel, fmaps, x.shape[1].value],
        gain=gain,
        use_wscale=use_wscale,
        fan_in=(kernel**2) * x.shape[1].value)
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    w = spectral_norm(w, epsilon=epsilon)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(
        x, w, os, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')


#----------------------------------------------------------------------------
# Box filter downscaling layer.


def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(
            x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW'
        )  # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True


#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.


def conv2d_downscale2d(x,
                       fmaps,
                       kernel,
                       gain=np.sqrt(2),
                       use_wscale=False,
                       epsilon=1e-8):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight(
        [kernel, kernel, x.shape[1].value, fmaps],
        gain=gain,
        use_wscale=use_wscale)
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = spectral_norm(w, epsilon=epsilon)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(
        x, w, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')


#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(
            tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


#----------------------------------------------------------------------------
# Minibatch standard deviation.


def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[
            0])  # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape  # [NCHW]  Input shape.
        y = tf.reshape(
            x, [group_size, -1, s[1], s[2],
                s[3]])  # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)  # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(
            y, axis=0, keepdims=True)  # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(
            tf.square(y), axis=0)  # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)  # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(
            y, axis=[1, 2, 3],
            keepdims=True)  # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)  # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2],
                        s[3]])  # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)  # [NCHW]  Append as new fmap.


def spectral_norm(w, iteration=1, epsilon=1e-8):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(
        "u", [1, w_shape[-1]],
        initializer=tf.truncated_normal_initializer(),
        trainable=False)

    u = tf.cast(u, w.dtype)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_, epsilon=epsilon)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_, epsilon=epsilon)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    u = u_hat
    w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def l2_norm(v, epsilon=1e-8):
    return v / (tf.reduce_sum(v**2)**0.5 + epsilon)


def attention(x, ch, scope='attention', epsilon=1e-8):
    res_log = np.log2(ch)
    if res_log > 5:
        _x = x
        with tf.variable_scope(scope, reuse=False):
            if res_log < 8:
                fact = 2
                if res_log < 7:
                    fact = 4
                if res_log < 6:
                    fact = 4
                if res_log <= 5:
                    fact = 8
                print("Downsampling x {} fact {}".format(x.shape, fact))
                _x = downscale2d(x, factor=fact)
            with tf.variable_scope('f', reuse=False):
                f = conv2d(
                    _x, ch // 8, kernel=1, epsilon=epsilon)  # [bs, h, w, c']
            with tf.variable_scope('g', reuse=False):
                g = conv2d(
                    _x, ch // 8, kernel=1, epsilon=epsilon)  # [bs, h, w, c']
            with tf.variable_scope('h', reuse=False):
                h = conv2d(_x, ch, kernel=1, epsilon=epsilon)  # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(
                hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s, axis=-1)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable(
                "gamma", [1], initializer=tf.constant_initializer(0.0))

            gamma = tf.cast(gamma, x.dtype)
            o = tf.reshape(o, shape=tf.shape(_x))  # [bs, h, w, C]
            if res_log < 8:
                # print("Upsampling x {} fact {}".format(o.shape, fact))
                o = upscale2d(o, factor=fact)
            x = gamma * o + x

        return x


def hw_flatten(x):
    x_shape = x.get_shape().as_list()
    shape_tensor = [tf.shape(x)[0], x_shape[1], np.prod(x_shape[2:])]
    x = tf.reshape(x, shape=shape_tensor)
    x = tf.transpose(x, perm=[0, 2, 1])
    return x


#----------------------------------------------------------------------------
# Generator network used in the paper.


def G_paper_att(
        latents_in,  # First input: Latent vectors [minibatch, latent_size].
        labels_in,  # Second input: Labels [minibatch, label_size].
        num_channels=1,  # Number of output color channels. Overridden based on dataset.
        resolution=32,  # Output resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        latent_size=None,  # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
        normalize_latents=True,  # Normalize latent vectors before feeding them to the network?
        use_wscale=True,  # Enable equalized learning rate?
        use_pixelnorm=True,  # Enable pixelwise feature vector normalization?
        pixelnorm_epsilon=1e-8,  # Constant epsilon for pixelwise feature vector normalization.
        use_leakyrelu=True,  # True = leaky ReLU, False = ReLU.
        dtype='float32',  # Data type to use for activations and outputs.
        fused_scale=True,  # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
        structure=None,  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        **kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)

    def PN(x):
        return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x

    if latent_size is None: latent_size = nf(0)
    if structure is None:
        structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(
        tf.get_variable('lod', initializer=np.float32(0.0), trainable=False),
        dtype)

    # Building blocks.
    def block(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == 2:  # 4x4
                if normalize_latents:
                    x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(
                        x,
                        fmaps=nf(res - 1) * 16,
                        gain=np.sqrt(2) / 4,
                        use_wscale=use_wscale,
                        epsilon=pixelnorm_epsilon
                    )  # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res - 1), 4, 4])
                    x = PN(act(apply_bias(x)))
                with tf.variable_scope('Conv'):
                    x = PN(
                        act(
                            apply_bias(
                                conv2d(
                                    x,
                                    fmaps=nf(res - 1),
                                    kernel=3,
                                    use_wscale=use_wscale,
                                    epsilon=pixelnorm_epsilon))))
            else:  # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = PN(
                            act(
                                apply_bias(
                                    upscale2d_conv2d(
                                        x,
                                        fmaps=nf(res - 1),
                                        kernel=3,
                                        use_wscale=use_wscale,
                                        epsilon=pixelnorm_epsilon))))
                else:
                    x = upscale2d(x)
                    with tf.variable_scope('Conv0'):
                        x = PN(
                            act(
                                apply_bias(
                                    conv2d(
                                        x,
                                        fmaps=nf(res - 1),
                                        kernel=3,
                                        use_wscale=use_wscale,
                                        epsilon=pixelnorm_epsilon))))
                with tf.variable_scope('Conv1'):
                    x = PN(
                        act(
                            apply_bias(
                                conv2d(
                                    x,
                                    fmaps=nf(res - 1),
                                    kernel=3,
                                    use_wscale=use_wscale,
                                    epsilon=pixelnorm_epsilon))))
            # print("Res: {}".format(res))
            # print("Nf: {}".format(nf(res - 1)))
            x = attention(x, nf(res - 1))
            return x

    def torgb(x, res):  # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(
                conv2d(
                    x,
                    fmaps=num_channels,
                    kernel=1,
                    gain=1,
                    use_wscale=use_wscale,
                    epsilon=pixelnorm_epsilon))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':

        def grow(x, res, lod):
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > 2:                img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0:
                img = cset(img, (lod_in < lod),
                           lambda: grow(y, res + 1, lod - 1))
            return img()

        images_out = grow(combo_in, 2, resolution_log2 - 2)

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out


#----------------------------------------------------------------------------
# Discriminator network used in the paper.


def D_paper(
        images_in,  # Input: Images [minibatch, channel, height, width].
        num_channels=1,  # Number of input color channels. Overridden based on dataset.
        resolution=32,  # Input resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        use_wscale=True,  # Enable equalized learning rate?
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        dtype='float32',  # Data type to use for activations and outputs.
        fused_scale=True,  # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
        structure=None,  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        epsilon=1e-8,
        **kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)

    if structure is None:
        structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(
        tf.get_variable('lod', initializer=np.float32(0.0), trainable=False),
        dtype)

    # Building blocks.
    def fromrgb(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(
                apply_bias(
                    conv2d(
                        x,
                        fmaps=nf(res - 1),
                        kernel=1,
                        use_wscale=use_wscale,
                        epsilon=epsilon)))

    def block(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3:  # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = act(
                        apply_bias(
                            conv2d(
                                x,
                                fmaps=nf(res - 1),
                                kernel=3,
                                use_wscale=use_wscale,
                                epsilon=epsilon)))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(
                            apply_bias(
                                conv2d_downscale2d(
                                    x,
                                    fmaps=nf(res - 2),
                                    kernel=3,
                                    use_wscale=use_wscale,
                                    epsilon=epsilon)))
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(
                            apply_bias(
                                conv2d(
                                    x,
                                    fmaps=nf(res - 2),
                                    kernel=3,
                                    use_wscale=use_wscale,
                                    epsilon=epsilon)))
                    x = downscale2d(x)
            else:  # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act(
                        apply_bias(
                            conv2d(
                                x,
                                fmaps=nf(res - 1),
                                kernel=3,
                                use_wscale=use_wscale,
                                epsilon=epsilon)))
                with tf.variable_scope('Dense0'):
                    x = act(
                        apply_bias(
                            dense(
                                x,
                                fmaps=nf(res - 2),
                                use_wscale=use_wscale,
                                epsilon=epsilon)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(
                        dense(
                            x,
                            fmaps=1 + label_size,
                            gain=1,
                            use_wscale=use_wscale,
                            epsilon=epsilon))
            return x

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':

        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0:
                x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res)
            y = lambda: x
            if res > 2:                y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()

        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out


#----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Generator network with FILM.


def G_film(
        latents_in,  # First input: Latent vectors [minibatch, latent_size].
        labels_in,  # Second input: Labels [minibatch, label_size].
        num_channels=1,  # Number of output color channels. Overridden based on dataset.
        resolution=32,  # Output resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        latent_size=None,  # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
        normalize_latents=True,  # Normalize latent vectors before feeding them to the network?
        use_wscale=True,  # Enable equalized learning rate?
        use_pixelnorm=True,  # Enable pixelwise feature vector normalization?
        pixelnorm_epsilon=1e-8,  # Constant epsilon for pixelwise feature vector normalization.
        use_leakyrelu=True,  # True = leaky ReLU, False = ReLU.
        weight_decay_film=1e-4,
        dtype='float32',  # Data type to use for activations and outputs.
        fused_scale=True,  # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
        structure='linear',  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        **kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)

    def PN(x):
        return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x

    if latent_size is None: latent_size = nf(0)
    if structure is None:
        structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(latents_in, dtype)
    text_embed = tf.cast(labels_in, dtype)
    lod_in = tf.cast(
        tf.get_variable('lod', initializer=np.float32(0.0), trainable=False),
        dtype)

    # Building blocks.
    def block(x, res, text_embed, weight_decay_film,
              **kwargs):  # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == 2:  # 4x4
                if normalize_latents:
                    x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(
                        x,
                        fmaps=nf(res - 1) * 16,
                        gain=np.sqrt(2) / 4,
                        use_wscale=use_wscale
                    )  # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res - 1), 4, 4])
                    x = apply_film(x, text_embed, weight_decay_film)
                    x = PN(act(x))
                with tf.variable_scope('Conv'):
                    x = conv2d(
                        x, fmaps=nf(res - 1), kernel=3, use_wscale=use_wscale)
                    x = apply_film(x, text_embed, weight_decay_film)
                    x = PN(act(x))
            else:  # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = upscale2d_conv2d(
                            x,
                            fmaps=nf(res - 1),
                            kernel=3,
                            use_wscale=use_wscale)
                        x = apply_film(x, text_embed, weight_decay_film)
                        x = PN(act(x))
                else:
                    x = upscale2d(x)
                    with tf.variable_scope('Conv0'):
                        x = conv2d(
                            x,
                            fmaps=nf(res - 1),
                            kernel=3,
                            use_wscale=use_wscale)
                        x = apply_film(x, text_embed, weight_decay_film)
                        x = PN(act(x))
                with tf.variable_scope('Conv1'):
                    print("Before conv2d")
                    print(x)
                    x = conv2d(
                        x, fmaps=nf(res - 1), kernel=3, use_wscale=use_wscale)
                    print("Before apply film")
                    print(x)
                    x = apply_film(x, text_embed, weight_decay_film)
                    print(x)
                    x = PN(act(x))
            print("Before attention: {}".format(x))
            if nf(res - 1) > 7:
                x = attention(x, nf(res - 1))
            print("before return block {}: {}".format(res, x))
            return x

    def torgb(x, res):  # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            print("in torgb")
            print(x)
            return apply_bias(
                conv2d(
                    x,
                    fmaps=num_channels,
                    kernel=1,
                    gain=1,
                    use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        text_embed, embedding_kl_loss = embed_condition(
            text_embed, fmaps=latent_size)
        tf.add_to_collection(
            name=tf.GraphKeys.REGULARIZATION_LOSSES, value=embedding_kl_loss)

        x = block(
            combo_in,
            2,
            text_embed=text_embed,
            weight_decay_film=weight_decay_film,
            **kwargs)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            print("RESOLUTION {}".format(res))
            print("Before last blocks")
            print(x)
            lod = resolution_log2 - res
            x = block(
                x,
                res,
                text_embed=text_embed,
                weight_decay_film=weight_decay_film,
                **kwargs)
            print("after block before to rgb")
            print(x)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':

        def grow(x, res, lod):
            y = block(
                x,
                res,
                text_embed=text_embed,
                weight_decay_film=weight_decay_film)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > 2:                img = cset(img, (lod_in > lod),
    lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod),
                      2 ** lod))
            if lod > 0:
                img = cset(img, (lod_in < lod),
                           lambda: grow(y, res + 1, lod - 1))
            return img()

        images_out = grow(combo_in, 2, resolution_log2 - 2)

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out


#----------------------------------------------------------------------------
# Generator network used in the paper.


def G_paper(
        latents_in,  # First input: Latent vectors [minibatch, latent_size].
        labels_in,  # Second input: Labels [minibatch, label_size].
        num_channels=1,  # Number of output color channels. Overridden based on dataset.
        resolution=32,  # Output resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        latent_size=None,  # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
        normalize_latents=True,  # Normalize latent vectors before feeding them to the network?
        use_wscale=True,  # Enable equalized learning rate?
        use_pixelnorm=True,  # Enable pixelwise feature vector normalization?
        pixelnorm_epsilon=1e-8,  # Constant epsilon for pixelwise feature vector normalization.
        use_leakyrelu=True,  # True = leaky ReLU, False = ReLU.
        dtype='float32',  # Data type to use for activations and outputs.
        fused_scale=True,  # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
        structure=None,  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        **kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)

    def PN(x):
        return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x

    if latent_size is None: latent_size = nf(0)
    if structure is None:
        structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu

    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(
        tf.get_variable('lod', initializer=np.float32(0.0), trainable=False),
        dtype)

    # Building blocks.
    def block(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == 2:  # 4x4
                if normalize_latents:
                    x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(
                        x,
                        fmaps=nf(res - 1) * 16,
                        gain=np.sqrt(2) / 4,
                        use_wscale=use_wscale
                    )  # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res - 1), 4, 4])
                    x = PN(act(apply_bias(x)))
                with tf.variable_scope('Conv'):
                    x = PN(
                        act(
                            apply_bias(
                                conv2d(
                                    x,
                                    fmaps=nf(res - 1),
                                    kernel=3,
                                    use_wscale=use_wscale))))
            else:  # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = PN(
                            act(
                                apply_bias(
                                    upscale2d_conv2d(
                                        x,
                                        fmaps=nf(res - 1),
                                        kernel=3,
                                        use_wscale=use_wscale))))
                else:
                    x = upscale2d(x)
                    with tf.variable_scope('Conv0'):
                        x = PN(
                            act(
                                apply_bias(
                                    conv2d(
                                        x,
                                        fmaps=nf(res - 1),
                                        kernel=3,
                                        use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(
                        act(
                            apply_bias(
                                conv2d(
                                    x,
                                    fmaps=nf(res - 1),
                                    kernel=3,
                                    use_wscale=use_wscale))))
            return x

    def torgb(x, res):  # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(
                conv2d(
                    x,
                    fmaps=num_channels,
                    kernel=1,
                    gain=1,
                    use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':

        def grow(x, res, lod):
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            if res > 2:                img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0:
                img = cset(img, (lod_in < lod),
                           lambda: grow(y, res + 1, lod - 1))
            return img()

        images_out = grow(combo_in, 2, resolution_log2 - 2)

    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out


#----------------------------------------------------------------------------
# Discriminator network used in the paper.


def D_paper(
        images_in,  # Input: Images [minibatch, channel, height, width].
        num_channels=1,  # Number of input color channels. Overridden based on dataset.
        resolution=32,  # Input resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        use_wscale=True,  # Enable equalized learning rate?
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        dtype='float32',  # Data type to use for activations and outputs.
        fused_scale=True,  # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
        structure=None,  # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        **kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)

    if structure is None:
        structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(
        tf.get_variable('lod', initializer=np.float32(0.0), trainable=False),
        dtype)

    # Building blocks.
    def fromrgb(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(
                apply_bias(
                    conv2d(
                        x, fmaps=nf(res - 1), kernel=1,
                        use_wscale=use_wscale)))

    def block(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3:  # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = act(
                        apply_bias(
                            conv2d(
                                x,
                                fmaps=nf(res - 1),
                                kernel=3,
                                use_wscale=use_wscale)))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(
                            apply_bias(
                                conv2d_downscale2d(
                                    x,
                                    fmaps=nf(res - 2),
                                    kernel=3,
                                    use_wscale=use_wscale)))
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(
                            apply_bias(
                                conv2d(
                                    x,
                                    fmaps=nf(res - 2),
                                    kernel=3,
                                    use_wscale=use_wscale)))
                    x = downscale2d(x)
            else:  # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act(
                        apply_bias(
                            conv2d(
                                x,
                                fmaps=nf(res - 1),
                                kernel=3,
                                use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(
                        apply_bias(
                            dense(x, fmaps=nf(res - 2),
                                  use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(
                        dense(
                            x,
                            fmaps=1 + label_size,
                            gain=1,
                            use_wscale=use_wscale))
            return x

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':

        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0:
                x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res)
            y = lambda: x
            if res > 2:                y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()

        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out


#----------------------------------------------------------------------------
