from tensorflow.keras import layers, backend


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
