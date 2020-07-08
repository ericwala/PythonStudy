# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:13:21 2020
事前放大成224*224，用catdogv22的MobileNetV2
原始碼
https://github.com/tensorflow/models/tree/master/research/object_detection

tutoraial
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
@author: user
"""


#from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# create the base pre-trained model
base_model = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False) 
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


train_path = r'D:\test\0518ultrasound\try224ss0630\data\train'
test_path = r'D:\test\0518ultrasound\try224ss0630\data\test'

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  #調到-1~1
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    directory=train_path, 
    target_size=(224,224),
    #classes=['dogs', 'cats'], 
    batch_size=1,
    shuffle=True) #我們有各768 train，跟下面的step有關，train的shuffle要開

train2_generator = train_datagen.flow_from_directory(
    directory=train_path, 
    target_size=(224,224),
    #classes=['dogs', 'cats'], 
    batch_size=1,
    shuffle=False)

test_generator = test_datagen.flow_from_directory(
    directory=test_path, 
    target_size=(224,224),
    #classes=['dogs', 'cats'], 
    batch_size=1,
    shuffle=False) #我們有各324test，跟下面的step有關

#用fit_genorator就可以直接將生成器(genorator直接丟進去)
model.fit_generator(generator=train_generator, steps_per_epoch=1536,  #step乘上面的batch_size=資料量，但我用2000會overfit所以用200
                    epochs=40,verbose=1)

"""
print('\nTesting ------------')
loss, accuracy = model.evaluate_generator(generator=train_generator,steps=384,verbose=1)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
"""



print('')
print('for train_data')
Y_pred_tr = model.predict_generator(train_generator,steps=1536,verbose=1)
y_pred_tr = np.argmax(Y_pred_tr, axis=1)
print('Confusion Matrix')
print(confusion_matrix(train_generator.classes, y_pred_tr))
print('Classification Report')
target_names = ['0', '1']
print(classification_report(train_generator.classes, y_pred_tr, target_names=target_names))



print('')
print('for train2_data')
Y2_pred_tr = model.predict_generator(train2_generator,steps=1536,verbose=1)
y2_pred_tr = np.argmax(Y2_pred_tr, axis=1)
print('Confusion Matrix')
print(confusion_matrix(train2_generator.classes, y2_pred_tr))
print('Classification Report')
target_names = ['0', '1']
print(classification_report(train2_generator.classes, y2_pred_tr, target_names=target_names))

print('')
print('for test_data')
Y_pred = model.predict_generator(test_generator,steps=648,verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['0', '1']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))



model.save('res_500_samegr_samenum_224_0630.h5')