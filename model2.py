
# coding: utf-8

# In[ ]:


# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
import tensorflow as tf
#from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16, preprocess_input
#from tensorflow.contrib.keras.python.keras.applications.vgg19 import VGG19, preprocess_input
#from tensorflow.contrib.keras.python.keras.applications.xception import Xception, preprocess_input 
#from tensorflow.contrib.keras.python.keras.applications.resnet50 import ResNet50, preprocess_input
#from tensorflow.contrib.keras.python.keras.applications.inception_v3 import InceptionV3, preprocess_input
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


# In[3]:


# path to training dataset
train_labels = sorted(os.listdir(train_path))


# In[4]:


# variables to hold features and labels
features = []
labels   = []


# In[5]:


image_files = glob.glob('jpg/*.jpg')


# In[6]:


label = 0
i = 0
j = 80


# In[7]:


for x in range(1, 18):
    for y in range(0, 80):
        labels.append(label)
    label += 1


# In[8]:


from tensorflow.contrib.keras.python.keras.models import Sequential


# In[9]:


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


# In[10]:


# loop over all the labels in the folder
for label in train_labels:
    cur_path = train_path + "/" + label
    for image_path in glob.glob(cur_path):
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        features.append(flat)


# In[11]:


# encode the labels using LabelEncoder
targetNames = np.unique(labels)
le = LabelEncoder()
le_labels = le.fit_transform(labels)


# In[12]:


# save features and labels 
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))


# In[13]:


h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))


# In[14]:


h5f_data.close()
h5f_label.close()


# In[1]:


from scipy import io


# In[2]:


data_splits = io.loadmat('datasplits.mat')


# In[10]:


type(data_splits['trn1'][0])


# In[17]:


trainData = []
testData = []
trainLabels = []
testLabels = []


# In[18]:


for i in data_splits['trn1'][0]:
    trainData.append(features[i])
    trainLabels.append(labels[i])

for i in data_splits['tst1'][0]:
    testData.append(features[i])
    testLabels.append(labels[i])


# In[19]:


trainData = np.array(trainData)
trainLabels = np.array(trainLabels)
testData = np.array(testData)
testLabels = np.array(testLabels)


# In[20]:


print(trainData.shape)
print(testData.shape)
print(trainLabels.shape)
print(testLabels.shape)


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[22]:


# use logistic regression as the model
model = LogisticRegression()


# In[23]:


model.fit(trainData, trainLabels)


# In[24]:


# use rank-1 and rank-5 predictions
f = open('output/flowers_17/xception/results.txt', "w")
rank_1 = 0
rank_5 = 0


# In[25]:


# loop over test data
for (label, features) in zip(testLabels, testData):
    # predict the probability of each class label and 
    # take the top-5 class labels
    predictions = model.predict_proba(np.atleast_2d(features))[0]
    predictions = np.argsort(predictions)[::-1][:5]
    
    # rank-1 prediction increment
    if label == predictions[0]:
        rank_1 += 1
        
    # rank-5 prediction increment
    if label in predictions:
        rank_5 += 1


# In[26]:


# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100


# In[27]:


# write the accuracies to file
f.write("rank-1: {:.2f}%\n".format(rank_1))
f.write("rank-5: {:.2f}%\n\n".format(rank_5))


# In[28]:


# evaluate the model of test data
preds = model.predict(testData)


# In[29]:


# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()


# In[30]:


import pickle


# In[31]:


# dump classifier to file
with open('output/flowers_17/xception/classifier.cpickle', "wb") as f:
    pickle.dump(model, f)


# In[32]:


class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
			   "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
			   "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
			   "windflower", "pansy"]


# In[33]:


import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[34]:


# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm, 
            annot=True,
            cmap="Set2")
plt.show()


# In[35]:


import glob
from scipy.ndimage import imread


# In[45]:


image_files = sorted(glob.glob('jpg/*.jpg'))


# In[ ]:


i = 339
print(data_splits['tst1'][0][i])
img_array = imread(image_files[data_splits['tst1'][0][i]])
pred = model.predict(testData[i])
print(pred)
plt.imshow(img_array)
plt.show()
print("예측 : ", class_names[pred[0]], ", 정답 : " , class_names[testLabels[i]])


# In[ ]:




