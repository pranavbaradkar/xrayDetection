# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 07:15:18 2019

@author: prana
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image 
from keras.models import Sequential  
from keras.models import load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.datasets import imdb
from keras import backend as K
import cv2

j=0
i=0
model = load_model('E:\sample4_CNN.h5')
#model.save('E:\sample_CNN.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


for i in range (20):
    img = cv2.imread(r'C:/Users/prana/positive/pic{:}.png'.format(i))
    img = cv2.resize(img,(150,150))
    img = np.reshape(img,[1,150,150,3])



    prediction = model.predict_classes(img)

    if  prediction [0][0] == 1:
        predict = 'positive'
    else:
        predict = 'negative'
    print(predict)
       


