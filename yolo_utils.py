import tensorflow as tf
from tensorflow.keras import layers, backend
from tensorflow.python.ops.gen_array_ops import shape
from layers import conv_block
from utillity import xywh_to_x1y1x2y2, bbox_iou, bbox_giou
import numpy as np


def get_boxes(pred, anchors, classes, strides, xyscale):
    # (batch_size, grid_size, grid_size, 3, 5+classes)
    pred = layers.Reshape((pred.shape[1], pred.shape[1], anchors.shape[0], 5 + classes))(pred)
    # (?, 52, 52, 3, 2) (?, 52, 52, 3, 2) (?, 52, 52, 3, 1) (?, 52, 52, 3, 80)
    box_xy, box_wh, obj_prob, class_prob = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)  # (?, 52, 52, 3, 2)
    obj_prob = tf.sigmoid(obj_prob)  # (?, 52, 52, 3, 1)
    class_prob = tf.sigmoid(class_prob)  # (?, 52, 52, 3, 80)
    pred_box_xywh = tf.concat((box_xy, box_wh), axis=-1)  # (?, 52, 52, 3, 4)

    grid = tf.meshgrid(tf.range(pred.shape[1]), tf.range(pred.shape[1]))  # (52, 52) (52, 52)
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # (52, 52, 1, 2)
    grid = tf.cast(grid, dtype=tf.float32)

    box_xy = ((box_xy * xyscale) - 0.5 * (xyscale - 1) + grid) * strides  # (?, 52, 52, 1, 4)

    box_wh = tf.exp(box_wh) * anchors  # (?, 52, 52, 3, 2)
    box_x1y1 = box_xy - box_wh / 2  # (?, 52, 52, 3, 2)
    box_x2y2 = box_xy + box_wh / 2  # (?, 52, 52, 3, 2)
    pred_box_x1y1x2y2 = tf.concat([box_x1y1, box_x2y2], axis=-1)  # (?, 52, 52, 3, 4)

    return [pred_box_x1y1x2y2, obj_prob, class_prob, pred_box_xywh]


def yolo_detector(prediction, anchors, classes, strides, xyscale):
    small = get_boxes(prediction[0], anchors[0, :, :], classes, strides[0], xyscale[0])
    medium = get_boxes(prediction[1], anchors[1, :, :], classes, strides[1], xyscale[1])
    large = get_boxes(prediction[2], anchors[2, :, :], classes, strides[2], xyscale[2])

    return [*small, *medium, *large]

def yolo_detector_light(prediction, anchors, classes, strides, xyscale):

    medium = get_boxes(prediction[0], anchors[0, :, :], classes, strides[0], xyscale[0])
    large = get_boxes(prediction[1], anchors[1, :, :], classes, strides[1], xyscale[1])
    
    return [*medium, *large]

def yolo_postulate(conv_output, anchors, stride, num_class):
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    feature_map_size = conv_shape[1]
    anchor_per_scale = anchors.shape[0] # change able
    conv_output = tf.reshape(conv_output,
                             (batch_size, feature_map_size, feature_map_size, anchor_per_scale, 5 + num_class))

    raw_txty = conv_output[..., 0:2]
    raw_twth = conv_output[..., 2:4]
    raw_conf = conv_output[..., 4:5]
    raw_prob = conv_output[..., 5:]

    y = tf.tile(tf.range(feature_map_size, dtype=tf.int32)[:, tf.newaxis], [1, feature_map_size])
    x = tf.tile(tf.range(feature_map_size, dtype=tf.int32)[tf.newaxis, :], [feature_map_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(
        raw_txty) + xy_grid) * stride  # pengalian terhadap stride membuat titik xy realtif terhadap input image size
    pred_wh = (tf.exp(raw_twth) * anchors)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(raw_conf)
    pred_prob = tf.sigmoid(raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def yolo_loss_layer(conv, pred, label, bboxes, stride, classes, iou_loss_thresh):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor_size = pred.shape[3] #bermasalah
    input_size = output_size * stride

    conv = tf.reshape(conv, (batch_size, output_size, output_size, anchor_size, 5 + classes))

    raw_class_prob = conv[..., 5:]
    raw_conf = conv[..., 4:5]

    pred_xywh = pred[..., 0:4]
    pred_conf = pred[..., 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_class_prob = label[:, :, :, :, 5:]
    #center lose
    ciou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)

    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)
    # prob loss
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_class_prob, logits=raw_class_prob)
    # conf loss
    expand_pred_xywh = pred_xywh[:, :, :, :, np.newaxis, :]
    expand_bboxes = bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]

    iou = bbox_iou(expand_pred_xywh, expand_bboxes)
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_thresh, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=raw_conf)
    )

    ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return ciou_loss, conf_loss, prob_loss


def yolo_loss(args, classes, iou_loss_thresh, anchors):
    conv_sbbox = args[0]  # (None, 52, 52, 75)
    conv_mbbox = args[1]  # (None, 26, 26, 75)
    conv_lbbox = args[2]  # (None, 13, 13, 75)


    label_sbbox = args[3]  # (None, 52, 52, 3, 25)
    label_mbbox = args[4]  # (None, 26, 26, 3, 25)
    label_lbbox = args[5]  # (None, 13, 13, 3, 25)
    true_boxes = args[6]  # (None, 100, 4)

    pred_sbbox = yolo_postulate(conv_sbbox, anchors[0], 8, classes)  # (None, None, None, 3, 25)
    pred_mbbox = yolo_postulate(conv_mbbox, anchors[1], 16, classes)  # (None, None, None, 3, 25)
    pred_lbbox = yolo_postulate(conv_lbbox, anchors[2], 32, classes)  # (None, None, None, 3, 25)

    sbbox_ciou_loss, sbbox_conf_loss, sbbox_prob_loss = yolo_loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_boxes, 8,
                                                                   classes, iou_loss_thresh)
    mbbox_ciou_loss, mbbox_conf_loss, mbbox_prob_loss = yolo_loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_boxes, 16,
                                                                   classes, iou_loss_thresh)
    lbbox_ciou_loss, lbbox_conf_loss, lbbox_prob_loss = yolo_loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_boxes, 32,
                                                                   classes, iou_loss_thresh)

    ciou_loss = (lbbox_ciou_loss + sbbox_ciou_loss + mbbox_ciou_loss) * 3.54
    conf_loss = (lbbox_conf_loss + sbbox_conf_loss + mbbox_conf_loss) * 64.3
    prob_loss = (lbbox_prob_loss + sbbox_prob_loss + mbbox_prob_loss) * 1

    return ciou_loss + conf_loss + prob_loss



def yolo_head(inputs, number_of_class, anchors_size=3):
    conv_pred = layers.Conv2D(anchors_size * (number_of_class + 5), kernel_size=1, strides=1, padding='same',use_bias=False)(inputs)
    return conv_pred


def yolo_single_loss(args, number_of_class, iou_loss_thresh, anchors, stride):
    conv = args[0]
    label = args[1]
    true_boxes = args[2]

    predic = yolo_postulate(conv, anchors, stride, number_of_class)

    cio_loss, conf_loss, prob_loss = yolo_loss_layer(conv, predic, label, true_boxes, stride, number_of_class, iou_loss_thresh)
    cio_loss = cio_loss + 3.54
    conf_loss = conf_loss + 64.3
    prob_loss = prob_loss + 1

    return conf_loss + cio_loss + prob_loss

def yolo_loss_light(args, classes, iou_loss_thresh, anchors):
 
    conv_mbbox = args[0]  # (None, 26, 26, 75)
    conv_lbbox = args[1]  # (None, 13, 13, 75)

    label_mbbox = args[2]  # (None, 26, 26, 3, 25)
    label_lbbox = args[3]  # (None, 13, 13, 3, 25)
    true_boxes = args[4]  # (None, 100, 4)

    pred_mbbox = yolo_postulate(conv_mbbox, anchors[0], 16, classes)  # (None, None, None, 3, 25)
    pred_lbbox = yolo_postulate(conv_lbbox, anchors[1], 32, classes)  # (None, None, None, 3, 25)
 
    mbbox_ciou_loss, mbbox_conf_loss, mbbox_prob_loss = yolo_loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_boxes, 16, classes, iou_loss_thresh)
    lbbox_ciou_loss, lbbox_conf_loss, lbbox_prob_loss = yolo_loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_boxes, 32, classes, iou_loss_thresh)

    ciou_loss = (lbbox_ciou_loss + mbbox_ciou_loss) * 2.54
    conf_loss = (lbbox_conf_loss + mbbox_conf_loss) * 50.3
    prob_loss = (lbbox_prob_loss + mbbox_prob_loss) * 1

    return conf_loss + ciou_loss + prob_loss