
# coding: utf-8

# In[1]:


# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
import tensorflow as tf
#from vgg16 import VGG16, preprocess_input
#from vgg19 import VGG19, preprocess_input
#from tensorflow.contrib.keras.python.keras.applications.xception import Xception, preprocess_input 
#from resnet50 import ResNet50, preprocess_input
#from inception_v3 import InceptionV3, preprocess_input
from xception import Xception, preprocess_input
from tensorflow.contrib.keras.python.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.models import model_from_json
from tensorflow.contrib.keras.python.keras.layers import Dense, Input, BatchNormalization, Activation, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
import _pickle as cPickle

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import sys

import time


# In[2]:


# config variables
model_name = 'xception'
weights = 'imagenet'
include_top = 0
train_path = 'jpg'
features_path = 'output/flowers_17/xception/features.h5'
labels_path = 'output/flowers_17/xception/labels.h5'
test_size = 0.1
results = 'output/flowers_17/xception/results.txt'
classfier_file = 'output/flowers_17/xception/classifier.cpickle'


# In[3]:


# create the pretrained models 
# check for pretrained weight usage or not
# check for top layers to be included or not
if model_name == "vgg16":
	base_model = VGG16(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
	image_size = (224, 224)
elif model_name == "vgg19":
	base_model = VGG19(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
	image_size = (224, 224)
elif model_name == "resnet50":
	base_model = ResNet50(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
	image_size = (224, 224)
elif model_name == "inceptionv3":
	base_model = InceptionV3(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed9').output)
	image_size = (299, 299)
elif model_name == "xception":
	base_model = Xception(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
	image_size = (299, 299)
else:
	base_model = None


# In[ ]:


# path to training dataset
train_labels = sorted(os.listdir(train_path))

# variables to hold features and labels
features = []
labels   = []

label = 0
i = 0
j = 80

for x in range(1, 18):
    for y in range(0, 80):
        labels.append(label)
    label += 1


# In[4]:


# label 17
# enchinacea
# train 68 test 17

enchinacea_imgfiles = sorted(glob.glob('flower/Enchinacea/*.jpg'))
for _ in range(len(enchinacea_imgfiles)):
    labels.append(17)
    
# label 18
# Frangipani
# train 133 test 33

frangipani_imgfiles = sorted(glob.glob('flower/Frangipani/*.jpg'))
for _ in range(len(frangipani_imgfiles)):
    labels.append(18)
    
# label 19
# Ipomoea_pandurata
# train 86 test 21

ipomoea_pandurata_imgfiles = sorted(glob.glob('flower/Ipomoea_pandurata/*.jpg'))
for _ in range(len(ipomoea_pandurata_imgfiles)):
    labels.append(19)
    
# label 20
# mugunghwa
# train 66 test 16

mugunghwa_imgfiles = sorted(glob.glob('flower/mugunghwa/*.jpg'))
for _ in range(len(mugunghwa_imgfiles)):
    labels.append(20)
    
# label 21
# Nymphaea_odorata
# train 155 test 39

nymphaea_odorata_imgfiles = sorted(glob.glob('flower/Nymphaea_odorata/*.jpg'))
for _ in range(len(nymphaea_odorata_imgfiles)):
    labels.append(21)


# In[5]:


# encode the labels using LabelEncoder
targetNames = np.unique(labels)
le = LabelEncoder()
le_labels = le.fit_transform(labels)


# In[11]:


# loop over all the labels in the folder
for label in train_labels:
    cur_path = train_path + "/" + label
    for image_path in glob.glob(cur_path):
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        print(label + " complete")
        sys.stdout.flush()
        feature = model.predict(x)
        flat = feature.flatten()
        features.append(flat)

for img_file in enchinacea_imgfiles:
    img = image.load_img(img_file, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(img_file + " complete")
    feature = model.predict(x)
    flat = feature.flatten()
    features.append(flat)
    
for img_file in frangipani_imgfiles:
    img = image.load_img(img_file, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(img_file + " complete")
    feature = model.predict(x)
    flat = feature.flatten()
    features.append(flat)
    
for img_file in ipomoea_pandurata_imgfiles:
    img = image.load_img(img_file, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(img_file + " complete")
    feature = model.predict(x)
    flat = feature.flatten()
    features.append(flat)
    
for img_file in mugunghwa_imgfiles:
    img = image.load_img(img_file, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(img_file + " complete")
    feature = model.predict(x)
    flat = feature.flatten()
    features.append(flat)
    
for img_file in nymphaea_odorata_imgfiles:
    img = image.load_img(img_file, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(img_file + " complete")
    feature = model.predict(x)
    flat = feature.flatten()
    features.append(flat)


# In[49]:


# save features and labels 
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()


# In[ ]:




