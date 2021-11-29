import cv2
import yolo
import mobilenetyolo as myolo
import config
import os
from tensorflow.keras import models


weight_path = os.path.join('assets', 'model-mobilenet-yolo.55-97.82.h5')
val_notation = os.path.join('assets', 'validation.txt')
train_notation = os.path.join('assets', 'training.txt')
class_name = os.path.join('assets', 'class_name.txt')


cfg = {
  'image_size' : (608, 608, 3),
  'anchors' : [5, 8, 15, 28, 37, 61, 80, 125, 159, 230],
  'stride' : 32,
  'xyscale': 1.05,

  # Training
  'iou_loss_thresh': 0.5,
  'batch_size': 8,
  'num_gpu': 1,  # 2,

  # Inference
  'max_boxes': 100,
  'iou_threshold': 0.413,
  'score_threshold': 0.3,
}

# define a video capture object
vid = cv2.VideoCapture(0)
# model = yolo.Yolo(class_name, config.config, weight_path)
modellight = myolo.MNetYolo(cfg, class_name, weight_path)



while(True):

  ret, frame = vid.read()
  #print(frame.shape)
  frame = modellight.predict_raw(frame)
  cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.imwrite('test.jgp', frame)
    break

vid.release()
cv2.destroyAllWindows()

# matrix = cv2.imread('test.png')
# print(matrix)


# #frame = cv2.resize(frame, (416, 416))
# #frame = model.predict_raw(frame)

# cv2.imwrite('test.png', frame)
  
#     # Display the resulting frame
# cv2.imshow('frame', frame)


# # Destroy all the windows
# if cv2.waitKey(1) & 0xFF == ord('q'):
#   vid.release()
#   cv2.destroyAllWindows()