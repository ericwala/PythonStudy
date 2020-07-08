import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D,Dropout, Dense, Flatten
from keras.utils import np_utils
from keras.datasets import mnist
from keras.backend.tensorflow_backend import set_session
from PIL import Image
import os

def load_image(img_file, target_size=(224,224)):
    X = np.zeros((1, *target_size, 3))
    X[0, ] = np.asarray(tf.keras.preprocessing.image.load_img(
        img_file,
        target_size=target_size)
    )
    X = tf.keras.applications.mobilenet.preprocess_input(X)
    return X
def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

model = tf.keras.applications.mobilenet_v2.MobileNetV2(
  input_shape=(224, 224, 3),
  include_top=False,
  pooling='avg'
)


model.summary()