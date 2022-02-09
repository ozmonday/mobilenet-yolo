import tensorflow as tf
from tensorflow.keras import layers, backend, initializers



def conv(x, filters, kernel_size, downsampling=False, activation='leaky', batch_norm=True):
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))

    if downsampling:
        x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)  # top & left padding
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=not batch_norm,
                      # kernel_regularizer=regularizers.l2(0.0005),
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                      # bias_initializer=initializers.Zeros()
                      )(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    if activation == 'mish':
        x = mish(x)
    elif activation == 'leaky':
        x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(x, filters1, filters2, activation='leaky'):
    """
    :param x: input tensor
    :param filters1: num of filter for 1x1 conv
    :param filters2: num of filter for 3x3 conv
    :param activation: default activation function: leaky relu
    :return:
    """
    y = conv(x, filters1, kernel_size=1, activation=activation)
    y = conv(y, filters2, kernel_size=3, activation=activation)
    return layers.Add()([x, y])


def csp_block(x, residual_out, repeat, residual_bottleneck=False):
    """
    Cross Stage Partial Network (CSPNet)
    transition_bottleneck_dims: 1x1 bottleneck
    output_dims: 3x3
    :param x:
    :param residual_out:
    :param repeat:
    :param residual_bottleneck:
    :return:
    """
    route = x
    route = conv(route, residual_out, 1, activation="mish")
    x = conv(x, residual_out, 1, activation="mish")
    for i in range(repeat):
        x = residual_block(x,
                           residual_out // 2 if residual_bottleneck else residual_out,
                           residual_out,
                           activation="mish")
    x = conv(x, residual_out, 1, activation="mish")

    x = layers.Concatenate()([x, route])
    return x


def depth_wise_separable_convolution(inputs, pointwise_filters, depth_multiplier=1, strides=(1, 1), block_id=0):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    x = inputs if strides == (1, 1) else layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(inputs)
    # Depthwise
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
    # Separable
    x = layers.Conv2D(pointwise_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)


def conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1), block_id=0):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    x = layers.Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv_block_%d' % block_id)(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_block_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_block_%d_relu' % block_id)(x)


def inverted_res_block(inputs, expansion, strides, filters, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_filters = _make_divisible(filters, 8)
    prefix = 'block_{}_'.format(block_id)
    x = inputs
    x = conv_block(x, expansion * in_channels, kernel=(1, 1), block_id=block_id) if block_id else x
    x = x if strides == (1, 1) else layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(x)
    # Depthwise
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
    x = layers.Conv2D(pointwise_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)

    if in_channels == pointwise_filters and strides == (1, 1):
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
         new_v += divisor
    return new_v
