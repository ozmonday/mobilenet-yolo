
cfg = {
  'image_size' : (416, 416, 3),
  'anchors' : [4, 7, 9, 17, 17, 40, 31, 51, 49, 83, 82, 125, 98, 212, 175, 169, 194, 311],
  'strides' : [8, 16, 32],
  'xyscale': [1.2, 1.1, 1.05],

  # Training
  'iou_loss_thresh': 0.5,
  'batch_size': 8,
  'num_gpu': 1,  # 2,

  # Inference
  'max_boxes': 100,
  'iou_threshold': 0.413,
  'score_threshold': 0.3,
}

cfg_light = {
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
  'iou_threshold': 0.5,
  'score_threshold': 0.5,
}

cfg_new_light = { 
  'image_size' : (416, 416, 3),
  'anchors' : [ 3, 5, 7, 12, 12, 22, 19, 33, 26, 51, 52, 47, 39, 76, 54, 111, 98, 94, 79, 163, 157, 153, 113, 236, 224, 222, 183, 338 ],
  'strides' : [ 16, 32 ],
  'xyscale':[ 0.1, 0.05 ],
  'detector_count' : 2,
  'anchor_size_perdetector': 7,

  # Training
  'iou_loss_thresh': 0.5,
  'batch_size': 32,
  'num_gpu': 1,  # 2,

  # Inference
  'max_boxes': 100,
  'iou_threshold': 0.5,
  'score_threshold': 0.5,
}