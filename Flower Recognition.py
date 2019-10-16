import sys
from os.path import join
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import os

with tf.device("/device:GPU:0"):

    train_dir = 'data/train'
    validation_dir = 'data/validation'
    test_dir = 'test
    batch_size = 32
    target_size=(224, 224)

    train_datagen = ImageDataGenerator(featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,target_size=target_size,batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,target_size=target_size,batch_size=batch_size)

    test_generator = test_datagen.flow_from_directory(
        test_dir,target_size=target_size,batch_size=batch_size)

    vgg=VGG16(weights='imagenet',include_top=True)
    vgg.summary()

    model=Sequential()
    for layer in vgg.layers[:-1]:
        model.add(layer)

    for layer in model.layers[:]:
        layer.trainable=False

    model.add(Dense(102,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    history = model.fit_generator(
        generator = train_generator,
        steps_per_epoch=12978/32,
        epochs=100,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=50,
        callbacks=[EarlyStopping(min_delta=0.01)]
    )

    model.save("Vgg16.h5")

    submission=pd.read_csv('sample_submission.csv')
    submission.head()

    test_id=[]
    test_pred=[]

    for i in submission.image_id:
        img=cv2.resize(cv2.imread('test/'+str(i)+'.jpg'),(224,224))
        img=np.expand_dims(img,axis=0)
        test_id.append(i)
        test_pred.append(int(model.predict_classes(img)))

    final_submission=pd.DataFrame({'image_id':test_id,'category':test_pred})
    final_submission.head()

    final_submission.to_csv('final_submission.csv',index=False)

    

    
