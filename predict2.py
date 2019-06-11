# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:40:24 2019

@author: prana
"""

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images    -----   are these then grayscale (black and white)?
img_width, img_height = 150, 150

# load the model we saved
model = load_model('E:\sample4_CNN.h5')

# Get test image ready
for i in range (20):
    test_image = image.load_img(r'C:/Users/prana/positive/pic{:}.png'.format(i), target_size=(img_width, img_height))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    test_image = test_image.reshape(1,img_width, img_height,3)    # Ambiguity!
# Should this instead be: test_image.reshape(img_width, img_height, 3) ??

    result = model.predict(test_image, batch_size=1)
    result.astype(float)
    print (result)