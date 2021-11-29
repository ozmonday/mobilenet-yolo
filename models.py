from os import name
from layers import conv_block, depth_wise_separable_convolution, _make_divisible, inverted_res_block
from keras import models
from tensorflow.keras import layers, backend


def MobileNet(inputs):
    x = conv_block(inputs=inputs, filters=32, strides=(2, 2), block_id=0)
    x = depth_wise_separable_convolution(x, pointwise_filters=64, block_id=1)

    x = depth_wise_separable_convolution(
        x, pointwise_filters=128, block_id=2, strides=(2, 2))
    x = depth_wise_separable_convolution(x, pointwise_filters=128, block_id=3)

    x = depth_wise_separable_convolution(
        x, pointwise_filters=256, block_id=4, strides=(2, 2))
    route0 = depth_wise_separable_convolution(
        x, pointwise_filters=256, block_id=5)

    x = depth_wise_separable_convolution(
        route0, pointwise_filters=512, block_id=6, strides=(2, 2))
    x = depth_wise_separable_convolution(x, pointwise_filters=512, block_id=7)
    x = depth_wise_separable_convolution(x, pointwise_filters=512, block_id=8)
    x = depth_wise_separable_convolution(x, pointwise_filters=512, block_id=9)
    x = depth_wise_separable_convolution(x, pointwise_filters=512, block_id=10)
    route1 = depth_wise_separable_convolution(
        x, pointwise_filters=512, block_id=11)

    x = depth_wise_separable_convolution(
        route1, pointwise_filters=1024, block_id=12, strides=(2, 2))
    route2 = depth_wise_separable_convolution(
        x, pointwise_filters=1024, block_id=13)

    return models.Model(inputs, [route0, route1, route2], name='mobilenet')


def MobileNetV2(inputs):
    first_block_filters = _make_divisible(32, 8)
    x = conv_block(inputs=inputs, filters=first_block_filters,
                   kernel=(3, 3), strides=(2, 2), block_id=0)

    x = inverted_res_block(x, filters=16, strides=1, expansion=1, block_id=1)

    x = inverted_res_block(x, filters=24, strides=2, expansion=6, block_id=2)
    x = inverted_res_block(x, filters=24, strides=1, expansion=6, block_id=3)

    x = inverted_res_block(x, filters=32, strides=2, expansion=6, block_id=4)
    x = inverted_res_block(x, filters=32, strides=1, expansion=6, block_id=5)
    x = inverted_res_block(x, filters=32, strides=1, expansion=6, block_id=6)

    x = inverted_res_block(x, filters=64, strides=2, expansion=6, block_id=7)
    x = inverted_res_block(x, filters=64, strides=1, expansion=6, block_id=8)
    x = inverted_res_block(x, filters=64, strides=1, expansion=6, block_id=9)
    x = inverted_res_block(x, filters=64, strides=1, expansion=6, block_id=10)
    print(x.shape)
    x = inverted_res_block(x, filters=96, strides=1, expansion=6, block_id=11)
    x = inverted_res_block(x, filters=96, strides=1, expansion=6, block_id=12)
    x = inverted_res_block(x, filters=96, strides=1, expansion=6, block_id=13)

    x = inverted_res_block(x, filters=160, strides=2, expansion=6, block_id=14)
    x = inverted_res_block(x, filters=160, strides=1, expansion=6, block_id=15)
    x = inverted_res_block(x, filters=160, strides=1, expansion=6, block_id=16)
    
    x = inverted_res_block(x, filters=320, strides=1, expansion=6, block_id=17)

    x = conv_block(x, 1280, kernel=(1, 1), block_id=18)
    print(x.shape)
    return models.Model(inputs, x, name='mobilenetv2')


    


def FPN(backbone_model, classes=1):
    large, medium, small = backbone_model.outputs

    route_input = small
    x = conv_block(inputs=small, filters=256, kernel=(
        1, 1), strides=(1, 1), block_id=1)
    x = layers.UpSampling2D()(x)

    route1 = conv_block(inputs=medium, filters=256,
                        kernel=(1, 1), strides=(1, 1), block_id=2)
    x = layers.Concatenate()([route1, x])

    x = conv_block(inputs=x, filters=256, kernel=(
        1, 1), strides=(1, 1), block_id=3)
    x = conv_block(inputs=x, filters=512, kernel=(
        3, 3), strides=(1, 1), block_id=4)
    x = conv_block(inputs=x, filters=256, kernel=(
        1, 1), strides=(1, 1), block_id=5)
    x = conv_block(inputs=x, filters=512, kernel=(
        3, 3), strides=(1, 1), block_id=6)
    x = conv_block(inputs=x, filters=256, kernel=(
        1, 1), strides=(1, 1), block_id=7)

    route1 = x
    x = conv_block(inputs=x, filters=256, kernel=(
        1, 1), strides=(1, 1), block_id=8)
    x = layers.UpSampling2D()(x)

    route0 = conv_block(inputs=large, filters=128,
                        kernel=(1, 1), strides=(1, 1), block_id=9)
    x = layers.Concatenate()([route0, x])

    x = conv_block(inputs=x, filters=128, kernel=(
        1, 1), strides=(1, 1), block_id=10)
    x = conv_block(inputs=x, filters=256, kernel=(
        3, 3), strides=(1, 1), block_id=11)
    x = conv_block(inputs=x, filters=128, kernel=(
        1, 1), strides=(1, 1), block_id=12)
    x = conv_block(inputs=x, filters=256, kernel=(
        3, 3), strides=(1, 1), block_id=13)
    x = conv_block(inputs=x, filters=128, kernel=(
        1, 1), strides=(1, 1), block_id=14)

    route0 = x
    x = conv_block(inputs=x, filters=256, kernel=(
        3, 3), strides=(1, 1), block_id=15)
    conv_sbbox = layers.Conv2D(
        3 * (classes + 5), 1, strides=1, padding='same', use_bias=True)(x)

    route0 = layers.ZeroPadding2D(((1, 0), (1, 0)))(route0)
    x = layers.Conv2D(256, 3, strides=2, padding='valid',
                      use_bias=True)(route0)
    x = layers.Concatenate()([x, route1])

    x = conv_block(inputs=x, filters=256, kernel=(
        1, 1), strides=(1, 1), block_id=16)
    x = conv_block(inputs=x, filters=512, kernel=(
        3, 3), strides=(1, 1), block_id=17)
    x = conv_block(inputs=x, filters=256, kernel=(
        1, 1), strides=(1, 1), block_id=18)
    x = conv_block(inputs=x, filters=512, kernel=(
        3, 3), strides=(1, 1), block_id=19)
    x = conv_block(inputs=x, filters=256, kernel=(
        1, 1), strides=(1, 1), block_id=20)

    route1 = x
    x = conv_block(inputs=x, filters=512, kernel=(
        3, 3), strides=(1, 1), block_id=21)
    conv_mbbox = layers.Conv2D(
        3 * (classes + 5), 1, strides=1, padding='same', use_bias=True)(x)

    route1 = layers.ZeroPadding2D(((1, 0), (1, 0)))(route1)
    x = layers.Conv2D(512, 3, strides=2, padding='valid',
                      use_bias=True)(route1)
    x = layers.Concatenate()([x, route_input])

    x = conv_block(inputs=x, filters=512, kernel=(
        1, 1), strides=(1, 1), block_id=22)
    x = conv_block(inputs=x, filters=1024, kernel=(
        3, 3), strides=(1, 1), block_id=23)
    x = conv_block(inputs=x, filters=512, kernel=(
        1, 1), strides=(1, 1), block_id=24)
    x = conv_block(inputs=x, filters=1024, kernel=(
        3, 3), strides=(1, 1), block_id=25)
    x = conv_block(inputs=x, filters=512, kernel=(
        1, 1), strides=(1, 1), block_id=26)

    x = conv_block(inputs=x, filters=1024, kernel=(
        3, 3), strides=(1, 1), block_id=27)
    conv_lbbox = layers.Conv2D(
        3 * (classes + 5), 1, strides=1, padding='same', use_bias=True)(x)

    return [conv_sbbox, conv_mbbox, conv_lbbox]
