from os import name
from matplotlib import pyplot
import numpy as np
from six import iteritems

import cv2
import yolo_utils as yolo
import utillity
import tensorflow as tf


from tensorflow.keras import layers, models, optimizers
from models import MobileNet, FPN


class Yolo(object):
    def __init__(self, class_name_path, config, weight_path=None):
        super().__init__()
        self.anchors = np.array(config['anchors']).reshape((3, 3, 2))
        self.image_size = config['image_size']
        self.class_name = [line.strip()
                           for line in open(class_name_path).readlines()]
        self.number_of_class = len(self.class_name)
        self.max_boxes = config['max_boxes']
        self.iou_loss_thresh = config['iou_loss_thresh']
        self.strides = config['strides']
        self.xyscale = config['xyscale']
        self.iou_threshold = config['iou_threshold']
        self.score_threshold = config['score_threshold']
        self.weight_path = weight_path

        self.build_model(load_pretrained=True if self.weight_path else False)

    def build_model(self, load_pretrained=True):
        input_layer = layers.Input(self.image_size)
        backbone = MobileNet(input_layer)
        output_layer = FPN(backbone, self.number_of_class)
        self.yolo_model = models.Model(input_layer, output_layer)

        if load_pretrained:
            self.yolo_model.load_weights(self.weight_path)
            print(f'load from {self.weight_path}')

        y_true = [
            # label small boxes
            layers.Input(shape=(52, 52, 3, (self.number_of_class + 5))),
            # label medium boxes
            layers.Input(shape=(26, 26, 3, (self.number_of_class + 5))),
            # label large boxes
            layers.Input(shape=(13, 13, 3, (self.number_of_class + 5))),
            # true bboxes
            layers.Input(shape=(self.max_boxes, 4)), 
        ]

        loss_list = layers.Lambda(yolo.yolo_loss, arguments={
                                  'classes': self.number_of_class, 'iou_loss_thresh': self.iou_loss_thresh, 'anchors': self.anchors})([*self.yolo_model.outputs, *y_true])
        self.training_model = models.Model(
            [self.yolo_model.input, *y_true], loss_list)

        yolo_output = yolo.yolo_detector(self.yolo_model.outputs, anchors=self.anchors,
                                         classes=self.number_of_class, strides=self.strides, xyscale=self.xyscale)
        nms = utillity.nms(yolo_output, input_shape=self.image_size, num_class=self.number_of_class,
                           iou_threshold=self.iou_threshold, score_threshold=self.score_threshold)
        self.inferance_model = models.Model(input_layer, nms)

        self.training_model.compile(optimizer=optimizers.Adam(
            learning_rate=1e-3), loss=lambda y_true, y_pred: y_pred)

    def predict(self, img_path, plot_img=True):
        img = utillity.open_image(img_path)
        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]))
        img = img / 255
        img_exp = np.expand_dims(img, axis=0)
        predic = self.inferance_model.predict(img_exp)
        df = utillity.get_detection_data(predic, img.shape)
        utillity.plot_bbox(img, df, plot_img)

    def predict_raw(self, frame):
        frame = cv2.resize(frame, self.image_size[:2])
        frame = frame /255
        frame_exp = np.expand_dims(frame, axis=0)
        predic = self.inferance_model(frame_exp)
        df = utillity.get_detection_data(predic, frame.shape)
        return utillity.draw_bbox(frame, df)




