import keras.models
from tensorflow import keras
import tensorflow as tf

import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as viz_utils
#from object_detection.utils import config_util

model= keras.models.load_model('datsetTraining/completeSavedModel')

def objectDetection(cap):
# cap = cv2.VideoCapture(0)
# width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    labels =[ 'Back Phone', 'Front Phone', 'Nut', 'Screw', 'Wallet']
    # while (True):
    ret, frame = cap.read()

    frame = cv2.resize(frame, (256, 256))
    frame = frame.reshape(-1,256,256,3)

    detections = model(frame)
    # print(detections)
    softmax = tf.nn.softmax(detections)
    index = tf.math.argmax(softmax, axis=-1)[0]
    return labels[index]
    # print(softmax)

    # frame = frame.reshape(256, 256, 3)
    # cv2.imshow('object detection', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    

    # cap.release()
    # # Destroy all the windows
    # cv2.destroyAllWindows()