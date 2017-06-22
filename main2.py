# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
import tensorflow as tf
#from vgg16 import VGG16, preprocess_input
#from vgg19 import VGG19, preprocess_input
#from resnet50 import ResNet50, preprocess_input
#from inception_v3 import InceptionV3, preprocess_input
#from xception import Xception, preprocess_input
from tensorflow.contrib.keras.python.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.models import model_from_json
from tensorflow.contrib.keras.python.keras.layers import Dense, Input, BatchNormalization, Activation, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
import _pickle as cPickle

# scikit-learn
from sklearn.linear_model import LogisticRegression

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
import pickle

import time

def Run(self, img_path, model_name):
       
    # config variables
    weights = 'imagenet'
    include_top = 0
    train_path = 'jpg'
    classfier_file = 'output/flowers_17/' + model_name + '/classifier.cpickle'
    
    # create the pretrained models 
    # check for pretrained weight usage or not
    # check for top layers to be included or not
    if model_name == "vgg16":
        from vgg16 import VGG16, preprocess_input
        base_model = VGG16(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        image_size = (224, 224)
    elif model_name == "vgg19":
        from vgg19 import VGG19, preprocess_input
        base_model = VGG19(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        image_size = (224, 224)
    elif model_name == "resnet50":
        from resnet50 import ResNet50, preprocess_input
        base_model = ResNet50(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        image_size = (224, 224)
    elif model_name == "inceptionv3":
        from inception_v3 import InceptionV3, preprocess_input
        base_model = InceptionV3(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed9').output)
        image_size = (299, 299)
    elif model_name == "xception":
        from xception import Xception, preprocess_input
        base_model = Xception(weights=weights)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        image_size = (299, 299)
    else:
        base_model = None
        
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array)
    feature = feature.flatten()
    with open(classfier_file, 'rb') as f:
        model2 = pickle.load(f)
    
    pred = model2.predict(feature)
    prob = model2.predict_proba(np.atleast_2d(feature))[0]
    
    return pred, prob[0]