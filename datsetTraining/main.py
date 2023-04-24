# #importing the libraries
import os
import cv2
import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as viz_utils
#from object_detection.utils import config_util

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

ds_training = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\\Users\\Nicolas\\Documents\\Dataset\\',
    labels='inferred',
    color_mode='rgb',
    batch_size=32,
    shuffle=True,
    seed=123,
    validation_split=0.1,
    image_size=(256, 256),
    subset="training",
)


ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\\Users\\Nicolas\\Documents\\Dataset\\',
    labels='inferred',
    color_mode='rgb',
    batch_size=32,
    shuffle=True,
    seed=123,
    validation_split=0.1,
    image_size=(256, 256),
    subset="validation",
)

resize_and_rescale = tf.keras.Sequential([
    layers.Rescaling(1. / 255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomZoom(0.1),
    layers.RandomRotation(0.5),
    #layers.RandomBrightness([-0.05, 0.05]),
    #layers.RandomContrast(0.1),
])

model = keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Input((256, 256, 3)),
    layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(512, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, input_shape=(256, 256, 3)),
    layers.Dense(512),
    layers.Dense(256),
    layers.Dense(128),
    layers.Dense(5),
    layers.Softmax()
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(ds_training, batch_size=64, epochs=50, verbose=2)

model.evaluate(ds_validation, batch_size=64, verbose=2)
model.summary()
model.save('completeSavedModel/', )

