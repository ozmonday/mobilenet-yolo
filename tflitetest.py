import tensorflow as tf
import os 

model_path = os.path.join('assets', 'model-mobilenet-yolo.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
