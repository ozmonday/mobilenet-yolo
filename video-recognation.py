import cv2

import config
import os
import tensorflow as tf 
import numpy as np
import utillity as utill
import yolo_utils
import time


model_path = os.path.join('assets', 'model1.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocessor(img_raw, config, interpreter):
  anchors = np.array(config['anchors']).reshape((3, 3, 2))
  
  img = img_raw /255
  img = cv2.resize(img, config['image_size'][:2])
  img = np.array(img, dtype='float32')
  img_exp = np.expand_dims(img, axis=0)
  interpreter.set_tensor(input_details[0]['index'], img_exp)
  interpreter.invoke()
  large_output = interpreter.get_tensor(output_details[0]['index'])
  medium_output = interpreter.get_tensor(output_details[1]['index'])
  small_output = interpreter.get_tensor(output_details[2]['index'])
  outputs = [large_output, medium_output, small_output]
  outputs = yolo_utils.yolo_detector(outputs, anchors, 1, config['strides'], config['xyscale'])
  outputs = utill.nms(outputs, config['image_size'], 1, config['iou_threshold'], config['score_threshold'])
  boxes = utill.get_detection_data(outputs, img_raw.shape)
  
  return utill.draw_bbox(img_raw, boxes)


def preprocessor_light(img_raw, config, interpreter):
  anchors = np.array(config['anchors']).reshape((5, 2))
  
  img = img_raw /255
  img = cv2.resize(img, config['image_size'][:2])
  img = np.array(img, dtype='float32')
  img_exp = np.expand_dims(img, axis=0)
  interpreter.set_tensor(input_details[0]['index'], img_exp)
  interpreter.invoke()
  output = interpreter.get_tensor(output_details[0]['index'])
 
  output = yolo_utils.get_boxes(output, anchors, 1, config['stride'], config['xyscale'])
  output = utill.nms(output, config['image_size'], 1, config['iou_threshold'], config['score_threshold'])
  boxes = utill.get_detection_data(output, img_raw.shape)
  
  return utill.draw_bbox(img_raw, boxes)

# define a video capture object
vid = cv2.VideoCapture(0)
fps = vid.get(cv2.CAP_PROP_FPS)
print('Frame per Second : {0}'.format(fps))
num_frame = 1





while(True):
  start = time.time()
  ret, frame = vid.read()
  frame = preprocessor(frame, config.cfg, interpreter)
  end = time.time()

  second = end - start
  fps = num_frame/second

  cv2.putText(frame, "FPS : " + str(round(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
  cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.imwrite('test.jgp', frame)
    break

vid.release()
cv2.destroyAllWindows()