from os import name
from layers import conv_block, depth_wise_separable_convolution, _make_divisible, inverted_res_block, conv, residual_block, csp_block
from keras import models
from tensorflow.keras import layers


def DarkNet53(x):
    x = conv(x, 32, 3)
    x = conv(x, 64, 3, downsampling=True)

    for i in range(1):
        x = residual_block(x, 32, 64)
    x = conv(x, 128, 3, downsampling=True)

    for i in range(2):
        x = residual_block(x, 64, 128)
    x = conv(x, 256, 3, downsampling=True)

    for i in range(8):
        x = residual_block(x, 128, 256)
    route_1 = x
    x = conv(x, 512, 3, downsampling=True)

    for i in range(8):
        x = residual_block(x, 256, 512)
    route_2 = x
    x = conv(x, 1024, 3, downsampling=True)

    for i in range(4):
        x = residual_block(x, 512, 1024)

    return route_1, route_2, x


def CSPDarkNet53(input):
    x = conv(input, 32, 3)
    x = conv(x, 64, 3, downsampling=True)

    x = csp_block(x, residual_out=64, repeat=1, residual_bottleneck=True)
    x = conv(x, 64, 1, activation='mish')
    x = conv(x, 128, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=64, repeat=2)
    x = conv(x, 128, 1, activation='mish')
    x = conv(x, 256, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=128, repeat=8)
    x = conv(x, 256, 1, activation='mish')
    route0 = x
    x = conv(x, 512, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=256, repeat=8)
    x = conv(x, 512, 1, activation='mish')
    route1 = x
    x = conv(x, 1024, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=512, repeat=4)

    x = conv(x, 1024, 1, activation="mish")

    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)

    x = layers.Concatenate()([layers.MaxPooling2D(pool_size=13, strides=1, padding='same')(x),
                              layers.MaxPooling2D(pool_size=9, strides=1, padding='same')(x),
                              layers.MaxPooling2D(pool_size=5, strides=1, padding='same')(x),
                              x
                              ])
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    route2 = conv(x, 512, 1)
    return models.Model(input, [route0, route1, route2])



def MobileNet(inputs, light=True):
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
    
    if light:
        return models.Model(inputs, [route1, route2], name='mobilenet')
    
    return models.Model(inputs, [route0, route1, route2], name='mobilenet')



def MobileNetV2(inputs, light=True):
    first_block_filters = _make_divisible(32, 8)
    x = conv_block(inputs=inputs, filters=first_block_filters,
                   kernel=(3, 3), strides=(2, 2), block_id=0)

    x = inverted_res_block(x, filters=16, strides=1, expansion=1, block_id=1)

    x = inverted_res_block(x, filters=24, strides=2, expansion=6, block_id=2)
    x = inverted_res_block(x, filters=24, strides=1, expansion=6, block_id=3)

    x = inverted_res_block(x, filters=32, strides=2, expansion=6, block_id=4)
    x = inverted_res_block(x, filters=32, strides=1, expansion=6, block_id=5)
    route0 = inverted_res_block(x, filters=32, strides=1, expansion=6, block_id=6)

    x = inverted_res_block(route0, filters=64, strides=2, expansion=6, block_id=7)
    x = inverted_res_block(x, filters=64, strides=1, expansion=6, block_id=8)
    x = inverted_res_block(x, filters=64, strides=1, expansion=6, block_id=9)
    x = inverted_res_block(x, filters=64, strides=1, expansion=6, block_id=10)

    x = inverted_res_block(x, filters=96, strides=1, expansion=6, block_id=11)
    x = inverted_res_block(x, filters=96, strides=1, expansion=6, block_id=12)
    route1 = inverted_res_block(x, filters=96, strides=1, expansion=6, block_id=13)
    
    x = inverted_res_block(route1, filters=160, strides=2, expansion=6, block_id=14)
    x = inverted_res_block(x, filters=160, strides=1, expansion=6, block_id=15)
    x = inverted_res_block(x, filters=160, strides=1, expansion=6, block_id=16)
    
    x = inverted_res_block(x, filters=320, strides=1, expansion=6, block_id=17)
    route2 = conv_block(x, 1280, kernel=(1, 1), block_id=18)

    if light:
        return models.Model(inputs, [route1, route2], name='mobilenetv2')
    
    return models.Model(inputs, [route0, route1, route2], name='mobilenetv2')



def PANet(backbone_model, num_classes = 1):
    route0, route1, route2 = backbone_model.output

    route_input = route2
    x = conv(route2, 256, 1)
    x = layers.UpSampling2D()(x)
    route1 = conv(route1, 256, 1)
    x = layers.Concatenate()([route1, x])

    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)

    route1 = x
    x = conv(x, 128, 1)
    x = layers.UpSampling2D()(x)
    route0 = conv(route0, 128, 1)
    x = layers.Concatenate()([route0, x])

    x = conv(x, 128, 1)
    x = conv(x, 256, 3)
    x = conv(x, 128, 1)
    x = conv(x, 256, 3)
    x = conv(x, 128, 1)

    route0 = x
    x = conv(x, 256, 3)
    conv_sbbox = conv(x, 3 * (num_classes + 5), 1, activation=None, batch_norm=False)

    x = conv(route0, 256, 3, downsampling=True)
    x = layers.Concatenate()([x, route1])

    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)

    route1 = x
    x = conv(x, 512, 3)
    conv_mbbox = conv(x, 3 * (num_classes + 5), 1, activation=None, batch_norm=False)

    x = conv(route1, 512, 3, downsampling=True)
    x = layers.Concatenate()([x, route_input])

    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)

    x = conv(x, 1024, 3)
    conv_lbbox = conv(x, 3 * (num_classes + 5), 1, activation=None, batch_norm=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]
    


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


def FPN_light(backbone_model, classes=1, anchor_size=6):
    medium, small = backbone_model.outputs

    route_input = small
    x = conv_block(inputs=small, filters=256, kernel=(1, 1), strides=(1, 1), block_id=1)
    x = layers.UpSampling2D()(x)

    route1 = conv_block(inputs=medium, filters=256, kernel=(1, 1), strides=(1, 1), block_id=2)
    x = layers.Concatenate()([route1, x])

    x = conv_block(inputs=x, filters=256, kernel=(1, 1), strides=(1, 1), block_id=3)
    x = conv_block(inputs=x, filters=512, kernel=(3, 3), strides=(1, 1), block_id=4)
    x = conv_block(inputs=x, filters=256, kernel=(1, 1), strides=(1, 1), block_id=5)
    x = conv_block(inputs=x, filters=512, kernel=(3, 3), strides=(1, 1), block_id=6)
    x = conv_block(inputs=x, filters=256, kernel=(1, 1), strides=(1, 1), block_id=7)

    route1 = x
    x = conv_block(inputs=x, filters=512, kernel=(3, 3), strides=(1, 1), block_id=21)
    conv_mbbox = layers.Conv2D(anchor_size * (classes + 5), 1, strides=1, padding='same', use_bias=True)(x)

    route1 = layers.ZeroPadding2D(((1, 0), (1, 0)))(route1)
    x = layers.Conv2D(512, 3, strides=2, padding='valid', use_bias=True)(route1)
    x = layers.Concatenate()([x, route_input])

    x = conv_block(inputs=x, filters=512, kernel=(1, 1), strides=(1, 1), block_id=22)
    x = conv_block(inputs=x, filters=1024, kernel=(3, 3), strides=(1, 1), block_id=23)
    x = conv_block(inputs=x, filters=512, kernel=(1, 1), strides=(1, 1), block_id=24)
    x = conv_block(inputs=x, filters=1024, kernel=(3, 3), strides=(1, 1), block_id=25)
    x = conv_block(inputs=x, filters=512, kernel=(1, 1), strides=(1, 1), block_id=26)

    x = conv_block(inputs=x, filters=1024, kernel=(3, 3), strides=(1, 1), block_id=27)
    conv_lbbox = layers.Conv2D(anchor_size * (classes + 5), 1, strides=1, padding='same', use_bias=True)(x)

    return [conv_mbbox, conv_lbbox]



