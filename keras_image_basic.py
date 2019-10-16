import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications.resnet50 import preprocess_input
import os
from keras.applications import VGG16
#Load the VGG model
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import layers
from keras import Model

with tf.device("/device:GPU:0"):

    train = pd.read_csv('train.csv', dtype=str)
    test = pd.read_csv('test.csv', dtype=str)

    print(train.head(5))
    print(train.groupby(['category']).count())
    print("The training data frame: ",train.shape)

    image_size=500
    vgg_conv = VGG16(weights='/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(image_size, image_size, 3))

    for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

    model = Sequential()
    model.add(vgg_conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(102, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), 
                  loss = 'categorical_crossentropy', 
                  metrics = ['acc'])

    train['image_id']=train['image_id'].astype(str)
    train['image_id']=train['image_id']+".jpg"
    train['category']=train['category'].astype(str)
    
    print(train.head(10))

    test['image_id']=test['image_id'].astype(str)
    test['image_id']=test['image_id']+".jpg"

    print(test.head(10))

    datagen=ImageDataGenerator(rescale=1./255.,preprocessing_function=preprocess_input,validation_split=0.10)

    train_generator=datagen.flow_from_dataframe(
            dataframe=train,
            directory="train_data/",
            x_col="image_id",
            y_col="category",
            subset="training",
            batch_size=32,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(500,500))

    valid_generator=datagen.flow_from_dataframe(
            dataframe=train,
            directory="train/",
            x_col="image_id",
            y_col="category",
            subset="validation",
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(500,500))

    test_datagen=ImageDataGenerator(rescale=1./255.)
                        test_generator=test_datagen.flow_from_dataframe(
                        dataframe=test,
                        directory="test/",
                        x_col="image_id",
                        y_col="category",
                        batch_size=1,
                        seed=42,
                        shuffle=False,
                        class_mode=None,
                        target_size=(500,500))

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=5)

    submission=pd.read_csv("sample_submission.csv")
    print(submission.head(5))

    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    test_generator.reset()
    
    pred=model.predict_generator(test_generator,

    steps=STEP_SIZE_TEST,verbose=1)

    predicted_class_indices=np.argmax(pred,axis=1)

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    predictions = [labels[k] for k in predicted_class_indices]

    filenames=test_generator.filenames

    filenames=[f.split('.')[0] for f in filenames]

    results=pd.DataFrame({"image_id":filenames,"category":predictions})

    results = results.sort_values(by = ['image_id'], ascending = [True])

    print(results.head(10))

    results.to_csv("submission.csv",index=False)

    print('done')

    
    
