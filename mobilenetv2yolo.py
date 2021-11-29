import models
import yolo_utils
import numpy as np
import utillity
import tensorflow as tf

from tensorflow.keras import layers, models as md, optimizers


class MNetV2Yolo():
    def __init__(self, config, class_name_path, weight_path=None):
        self.image_size = config['image_size']
        self.iou_threshold = config['iou_threshold']
        self.iou_loss_thresh = config['iou_loss_thresh']
        self.stride = config['stride']
        self.xyscale = config['xyscale']
        self.score_threshold = config['score_threshold']
        self.max_boxes = config['max_boxes']
        self.class_name = [line.strip()
                           for line in open(class_name_path).readlines()]
        self.number_of_class = len(self.class_name)
        self.anchors = np.array(config['anchors']).reshape((5, 2))
        self.weight_path = weight_path

        self.build_model(load_pretrained=True if self.weight_path else False)

    def build_model(self, load_pretrained):
        inputs = layers.Input(self.image_size)
        feature_extractor = models.MobileNet(inputs)
        yolo_head = yolo_utils.yolo_head(
            feature_extractor.outputs[2], self.number_of_class, anchors_size=5)
        self.model = md.Model(inputs, yolo_head)

        if load_pretrained:
            self.model.load_weights(self.weight_path)
            print(f'load from {self.weight_path}')

        y_true = [
            layers.Input((19, 19, 5, (self.number_of_class + 5))),
            layers.Input((self.max_boxes, 4))
        ]

        loss_list = layers.Lambda(yolo_utils.yolo_single_loss, arguments={
            'number_of_class': self.number_of_class,
            'iou_loss_thresh': self.iou_loss_thresh,
            'anchors': self.anchors,
            'stride': self.stride})([self.model.outputs, *y_true])

        self.training_model = md.Model([self.model, *y_true], loss_list)

        model_output = yolo_utils.get_boxes(
            self.model.outputs, self.anchors, self.number_of_class, self.stride, self.xyscale)
        nms = utillity.nms(model_output, self.image_size,
                           self.number_of_class, self.iou_threshold, self.score_threshold)
        self.inferance_model = models.Model(inputs, nms)

        self.training_model.compile(optimizer=optimizers.Adam(
            learning_rate=1e-3), loss=lambda y_true, y_pred: y_pred)

    def predict(self, img_path, plot_img=True):
        img = utillity.open_image(img_path)
        img = tf.image.resize(img, (self.image_size[0], self.image_size[1]))
        img = img / 255
        img_exp = np.expand_dims(img, axis=0)
        predict = self.inferance_model.predict(img_exp)
        df = utillity.get_detection_data(predict)
        utillity.draw_bbox(img, df, plot_img)

    def fit(self, data_train, data_validation, initial_epoch, epochs, callback):
        self.training_model.fit(data_train, steps_per_epoch=len(
            data_train), validation_data=data_validation, epochs=epochs, initial_epoch=initial_epoch, callbacks=callback)


