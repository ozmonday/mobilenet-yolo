import os
import platform
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2

from tensorflow.keras.utils import Sequence
from matplotlib import image
from matplotlib import pyplot as plt
from matplotlib import patches
from tensorflow.keras import layers, backend



def xywh_to_x1y1x2y2(boxes):
    return tf.concat([boxes[..., :2] - boxes[..., 2:] * 0.5, boxes[..., :2] + boxes[..., 2:] * 0.5], axis=-1)


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # w * h
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # (x, y, w, h) -> (x0, y0, x1, y1)
    boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes2 = xywh_to_x1y1x2y2(boxes2)

    # coordinates of intersection
    top_left = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection_xy = tf.maximum(bottom_right - top_left, 0.0)

    intersection_area = intersection_xy[..., 0] * intersection_xy[..., 1]
    union_area = boxes1_area + boxes2_area - intersection_area

    return 1.0 * intersection_area / (union_area + tf.keras.backend.epsilon())


def bbox_giou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # w*h
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # (x, y, w, h) -> (x0, y0, x1, y1)
    boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes2 = xywh_to_x1y1x2y2(boxes2)

    top_left = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    intersection_xy = tf.maximum(bottom_right - top_left, 0.0)
    intersection_area = intersection_xy[..., 0] * intersection_xy[..., 1]

    union_area = boxes1_area + boxes2_area - intersection_area

    iou = 1.0 * intersection_area / (union_area + tf.keras.backend.epsilon())

    enclose_top_left = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_bottom_right = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])

    enclose_xy = enclose_bottom_right - enclose_top_left
    enclose_area = enclose_xy[..., 0] * enclose_xy[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


class DataGenerator(Sequence):
    def __init__(self, label_anotation, image_path, class_name_path, anchors, target_image_shape=(416, 416, 3),
                 batch_size=64, max_boxes=100, shuffle=True, num_stage=3, bbox_per_grid=3):

        self.label_anotation = label_anotation
        self.image_path = image_path
        self.class_name_path = class_name_path
        self.num_stage = num_stage
        self.bboxs_per_grid = bbox_per_grid
        self.num_classes = len([line.strip() for line in open(class_name_path).readlines()])
        self.max_boxes = max_boxes
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.target_image_shape = target_image_shape
        self.indexes = np.arange(len(self.label_anotation))
        self.anchors = np.array(anchors).reshape((num_stage*bbox_per_grid, 2))
        self.on_epoch_end()

    def __len__(self):
        '''number of batches per epoch'''
        return int(np.ceil(len(self.label_anotation) / self.batch_size))

    def __getitem__(self, index):

        idxs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        data = [self.label_anotation[i] for i in idxs]
        x, y_tensor, y_bbox = self.__data_generation(data)

        return [x, *y_tensor, y_bbox], np.zeros(len(data))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, label_anotation):
        x = np.empty((len(label_anotation), self.target_image_shape[0], self.target_image_shape[1],
                      self.target_image_shape[2]), dtype=np.float32)
        y_bbox = np.empty((len(label_anotation), self.max_boxes, 5), dtype=np.float32)

        for i, line in enumerate(label_anotation):
            img, boxes = get_data(line, image_path=self.image_path, target_image_shape=self.target_image_shape, max_boxes=self.max_boxes)
            x[i] = img
            y_bbox[i] = boxes

        y_tensor, y_true_boxes_xywh = pre_processing_true_bbox(y_bbox, self.target_image_shape[:2], self.anchors,
                                                             self.num_classes, self.num_stage, self.bboxs_per_grid)

        return x, y_tensor, y_true_boxes_xywh


def get_data(data, image_path, target_image_shape, max_boxes=100):
    if platform.system() == 'Windows':
        name = data['name'].replace("/", "\\")
    else:
        name = data['name']
    filepath = os.path.join(image_path, name)
    img = image.imread(filepath) / 255
    if img.shape != target_image_shape:
        img = tf.image.resize(img, (target_image_shape[0], target_image_shape[1]))
    boxes = np.array([[x['data'][0], x['data'][1], x['data'][2], x['data'][3], 0] for x in data['objects']])
    boxes_data = np.zeros((max_boxes, 5))

    if len(boxes) > 0:
        np.random.shuffle(boxes)
        boxes = boxes[:max_boxes]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * target_image_shape[0]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * target_image_shape[1]
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

        boxes_data[:len(boxes)] = boxes

    return img, boxes_data


def pre_processing_true_bbox(true_boxes, image_size, anchors, num_classes, num_stage, bbox_per_grid):
    anchor_mask = np.arange(0, num_stage*bbox_per_grid, dtype=int).reshape((num_stage, bbox_per_grid))
    anchor_mask = anchor_mask.tolist()
    true_boxes = np.array(true_boxes, dtype='float32')
    true_boxes_abs = np.array(true_boxes, dtype='float32')
    image_size = np.array(image_size, dtype='int32')

    true_boxes_xy = (true_boxes_abs[..., 0:2] + true_boxes_abs[..., 2:4]) // 2
    true_boxes_wh = true_boxes_abs[..., 2:4] - true_boxes_abs[..., 0:2]

    true_boxes[..., 0:2] = true_boxes_xy / image_size[::-1]
    true_boxes[..., 2:4] = true_boxes_wh / image_size[::-1]

    bs = true_boxes.shape[0]
    grid_size = [image_size // [8, 16, 32][-(s+1)] for s in range(num_stage)]
    #grid_size = [image_size // {0: 8, 1: 16, 2: 32}[s] for s in range(num_stage)]
    Y_true = [np.zeros((bs, grid_size[-(s+1)][0], grid_size[-(s+1)][1], bbox_per_grid, 5 + num_classes), dtype='float32') for s in
          range(num_stage)]
    #Y_true = [np.zeros((bs, grid_size[s][0], grid_size[s][1], bbox_per_grid, 5 + num_classes), dtype='float32') for s in range(num_stage)]
    Y_true_bbox_xywh = np.concatenate((true_boxes_xy, true_boxes_wh), axis=-1)

    anchors = np.expand_dims(anchors, 0)
    anchors_maxs = anchors / 2.
    anchors_mins = -anchors_maxs
    valid_mask = true_boxes_wh[..., 0] > 0

    for batch_index in range(bs):
        wh = true_boxes_wh[batch_index, valid_mask[batch_index]]
        if len(wh) == 0: continue
        wh = np.expand_dims(wh, -2)

        box_maxs = wh / 2.  # (# of bbox, 1, 2)
        box_mins = -box_maxs

        intersect_mins = np.maximum(box_mins, anchors_mins)
        intersect_maxs = np.minimum(box_maxs, anchors_maxs)
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = np.prod(intersect_wh, axis=-1)
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchors = np.argmax(iou, axis=-1)

        for box_index in range(len(wh)):
            best_anchor = best_anchors[box_index]
            for stage in range(num_stage):
                if best_anchor in anchor_mask[stage]:
                    x_offset = true_boxes[batch_index, box_index, 0] * grid_size[stage][1]
                    y_offset = true_boxes[batch_index, box_index, 1] * grid_size[stage][0]

                    grid_col = np.floor(x_offset).astype('int32')
                    grid_row = np.floor(y_offset).astype('int32')
                    anchor_idx = anchor_mask[stage].index(best_anchor)
                    class_idx = true_boxes[batch_index, box_index, 4].astype('int32')

                    Y_true[stage][batch_index, grid_row, grid_col, anchor_idx, :2] = true_boxes_xy[batch_index,
                                                                                     box_index, :]
                    Y_true[stage][batch_index, grid_row, grid_col, anchor_idx, 2:4] = true_boxes_wh[batch_index,
                                                                                      box_index, :]
                    Y_true[stage][batch_index, grid_row, grid_col, anchor_idx, 4] = 1

                    Y_true[stage][batch_index, grid_row, grid_col, anchor_idx, 5 + class_idx] = 1

    return Y_true, Y_true_bbox_xywh



def open_image(path, show=False):
    # if platform.system() == 'Windows':
    #     idx = name.replace('/', " ").split(" ")
    #     name = os.path.join(idx[0], idx[1])

    # path = os.path.join(path, name)
    img = image.imread(path)
    if show:
        plt.figure(figsize=(15, 15))
        plt.imshow(img, interpolation='nearest')
        plt.show()

    return img


def plot_bbox(img, detections, show_img=True):
    """
    Draw bounding boxes on the img.
    :param img: BGR img.
    :param detections: pandas DataFrame containing detections
    :param random_color: assign random color for each objects
    :param cmap: object colormap
    :param plot_img: if plot img with bboxes
    :return: None
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,15))
    ax1.imshow(img, interpolation='nearest')
    ax2.imshow(img, interpolation='nearest')

    for _, row in detections.iterrows():
        x1, y1, x2, y2, score, w, h = row.values
        rect = patches.Rectangle((int(x1), int(y1)), int(w), int(h), linewidth=3, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
        
    if show_img:
        plt.show()

def draw_bbox(raw_img, detections):

    raw_img = np.array(raw_img)
    scale = max(raw_img.shape[0:2]) / 416
    line_width = int(2 * scale)

    for _, row in detections.iterrows():
        x1, y1, x2, y2, score, w, h = row.values
        cv2.rectangle(raw_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), line_width)

    return raw_img
        



def get_detection_data(model_outputs, img_shape):
    """
    :param img: target raw image
    :param model_outputs: outputs from inference_model
    :param class_names: list of object class names
    :return:
    """

    num_bboxes = model_outputs[-1][0]
    boxes, scores, classes = [output[0][:num_bboxes] for output in model_outputs[:-1]]

    h = img_shape[0]
    w = img_shape[1]
    df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
    df[['x1', 'x2']] = (df[['x1', 'x2']] * w).astype('int64')
    df[['y1', 'y2']] = (df[['y1', 'y2']] * h).astype('int64')
    #df['class_name'] = np.array(class_names)[classes.astype('int64')]
    df['score'] = scores
    df['w'] = df['x2'] - df['x1']
    df['h'] = df['y2'] - df['y1']

    print(f'# of bboxes: {num_bboxes}')
    return df


def nms(model_ouputs, input_shape, num_class, iou_threshold=0.413, score_threshold=0.3):
    """
    Apply Non-Maximum suppression
    ref: https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    :param model_ouputs: yolo model model_ouputs
    :param input_shape: size of input image
    :return: nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
    """
    bs = tf.shape(model_ouputs[0])[0] #beach size
    boxes = tf.zeros((bs, 0, 4))
    confidence = tf.zeros((bs, 0, 1))
    class_probabilities = tf.zeros((bs, 0, num_class))

    for output_idx in range(0, len(model_ouputs), 4):
        output_xy = model_ouputs[output_idx]
        output_conf = model_ouputs[output_idx + 1]
        output_classes = model_ouputs[output_idx + 2]
        boxes = tf.concat([boxes, tf.reshape(output_xy, (bs, -1, 4))], axis=1)
        confidence = tf.concat([confidence, tf.reshape(output_conf, (bs, -1, 1))], axis=1)
        class_probabilities = tf.concat([class_probabilities, tf.reshape(output_classes, (bs, -1, num_class))], axis=1)

    scores = confidence * class_probabilities
    boxes = tf.expand_dims(boxes, axis=-2)
    boxes = boxes / input_shape[0]  # box normalization: relative img size
    print(f'nms iou: {iou_threshold} score: {score_threshold}')
    (nmsed_boxes,      # [bs, max_detections, 4]
     nmsed_scores,     # [bs, max_detections]
     nmsed_classes,    # [bs, max_detections]
     valid_detections  # [batch_size]
     ) = tf.image.combined_non_max_suppression(
        boxes=boxes,  # y1x1, y2x2 [0~1]
        scores=scores,
        max_output_size_per_class=100,
        max_total_size=100,  # max_boxes: Maximum nmsed_boxes in a single img.
        iou_threshold=iou_threshold,  # iou_threshold: Minimum overlap that counts as a valid detection.
        score_threshold=score_threshold,  # # Minimum confidence that counts as a valid detection.
    )
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


