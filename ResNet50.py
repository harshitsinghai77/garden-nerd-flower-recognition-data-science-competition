import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications import VGG19, ResNet50, InceptionV3
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import SGD, RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import json
import tensorflow as tf

with tf.device("/device:GPU:0"):
    train = pd.read_csv('train.csv')
    train.head()


    # In[4]:

    num_classes = len(set(train['category']))
    print(num_classes)
  
    vgg_model = VGG19(weights = "imagenet", include_top=False, input_shape = (256, 256, 3))
    
    for layer in vgg_model.layers[:-5]:
    	layer.trainable = False

    train_image = []
    for i in tqdm(range(train.shape[0])):
        img = image.load_img('train/'+train['image_id'][i].astype('str')+'.jpg', target_size=(256,256,3), grayscale=False)
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
    X = np.array(train_image)


    # In[ ]:


    y = train['category'].values.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    y = onehot_encoder.fit_transform(y)
    print(y.shape)


    train_data, val_data, train_labels, val_labels = train_test_split(X, y, random_state=42, test_size=0.2)


    # In[ ]:
    img_width, img_height = 256, 256

    def create_model_from_ResNet50():

        model = Sequential()

        model.add(vgg_model)
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(102, activation='softmax'))

        model.layers[0].trainable = False
        
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc']) # optimizer=RMSprop(lr=0.001)
        
        return model


    model_resnet = create_model_from_ResNet50()
    model_resnet.summary()

    # In[ ]:


    image_size = 224
    batch_size = 200


    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
    )

    train_generator = train_datagen.flow(
        train_data,
        train_labels,
        batch_size=batch_size
    )

    val_generator = val_datagen.flow(
        val_data,
        val_labels,
        batch_size=batch_size
    )

    checkpoint = keras.callbacks.ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    history = model_resnet.fit_generator(
        generator=train_generator, 
        steps_per_epoch=len(train_data)/32, 
        epochs=50, 
        validation_steps=len(val_data)/32,
        validation_data=val_generator,
        callbacks = [checkpoint, early])

    with open('Resnet_history.json', 'w') as f:
        json.dump(history.history, f)

    model_resnet.save_weights('50epochs_weights.h5')


    # In[ ]:


    model_resnet.save('flower_Resenet50_kaggle.h5')



