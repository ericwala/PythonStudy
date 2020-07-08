# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 22:11:40 2020

@author: user
"""


#from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing import image
import os

from keras.utils.np_utils import to_categorical

# create the base pre-trained model
base_model = ResNet50(input_shape=(50,50,3), weights='imagenet', include_top=False) 
#灰階還是照丟，train_generator那裏其實有color_mode可選，預設rgb，丟灰階他自動*3

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])

#====================================================
train_path_0 = r'D:\test\0518ultrasound\tryres0622\data\train\0'
train_path_1 = r'D:\test\0518ultrasound\tryres0622\data\train\1'
test_path_0 = r'D:\test\0518ultrasound\tryres0622\data\test\0'
test_path_1 = r'D:\test\0518ultrasound\tryres0622\data\test\1'
#=======================================================

train_cul = 0
train_label = np.array([])
test_cul = 0
test_label = np.array([])
label_0 = np.array([0])
label_1 = np.array([1])

for image_name in os.listdir(train_path_0):
    img = image.load_img(os.path.join(train_path_0, image_name), target_size=(50, 50))
    a = image.img_to_array(img)
    a = np.expand_dims(a, axis=0)
    a = preprocess_input(a)
    if train_cul == 0:
        train_data = a
        train_cul+=1
    else:
        train_data = np.append(train_data, a, axis=0)
    train_label = np.append(train_label, label_0)
    
for image_name in os.listdir(train_path_1):
    img = image.load_img(os.path.join(train_path_1, image_name), target_size=(50, 50))
    a = image.img_to_array(img)
    a = np.expand_dims(a, axis=0)
    a = preprocess_input(a)
    train_data = np.append(train_data, a, axis=0)
    train_label = np.append(train_label, label_1)
    

    
for image_name in os.listdir(test_path_0):
    img = image.load_img(os.path.join(test_path_0, image_name), target_size=(50, 50))
    a = image.img_to_array(img)
    a = np.expand_dims(a, axis=0)
    a = preprocess_input(a)
    if test_cul == 0:
        test_data = a
        test_cul+=1
    else:
        test_data = np.append(test_data, a, axis=0)
    test_label = np.append(test_label, label_0)   
    
for image_name in os.listdir(test_path_1):
    img = image.load_img(os.path.join(test_path_1, image_name), target_size=(50, 50))
    a = image.img_to_array(img)
    a = np.expand_dims(a, axis=0)
    a = preprocess_input(a)
    test_data = np.append(test_data, a, axis=0)
    test_label = np.append(test_label, label_1)

train_label = to_categorical(train_label,2)
test_label = to_categorical(test_label,2)

model.fit(train_data, train_label,epochs=300,batch_size=4,verbose=1)
model.save('new255.h5')