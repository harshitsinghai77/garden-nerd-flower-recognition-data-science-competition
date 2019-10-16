import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import load_model
from keras.preprocessing import image

model =  load_model('vgg16.h5')
model.summary()

test = pd.read_csv('test.csv')
print(test.head())

test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('test/'+test['image_id'][i].astype('str')+'.jpg', target_size=(224,224,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)

prediction = model.predict_classes(test)

sample = pd.read_csv('sample_submission.csv')
print(sample)
sample['category'] = prediction
print(sample)
sample.to_csv('base_model.csv')
