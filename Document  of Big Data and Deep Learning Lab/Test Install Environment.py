#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import urllib
import matplotlib.pyplot as plt

from platform import python_version
print('Python version: {}'.format(python_version()))

import cv2
print('OpenCV version: {}'.format(cv2.__version__))

import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))

import keras
print('Keras version: {}'.format(keras.__version__))
# a function to read image from an url
def url2image(url):
    #down load image, convert it to a numpy array, and then read it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype = 'uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
import sys
if sys.version_info[0]==3:
    from urllib.request import urlopen
else:
    from urllib import urlopen
    
#main
img_url = 'https://fortunedotcom.files.wordpress.com/2018/07/gettyimages-961697338.jpg'
bgr_img = url2image(img_url)

#BRG order is the default in OPENCV
plt.subplot(3,2,1)
plt.axis('off')
plt.title('Original image - BGR color')
plt.imshow(bgr_img)

plt.subplot(3,2,2)
plt.axis('off')
plt.title('Original image - CORRECT color')
plt.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
plt.subplot(3,2,3)
plt.axis('off')
plt.title('Gray image - No CMAP')
plt.imshow(gray_img)

plt.subplot(3,2,4)
plt.axis('off')
plt.title('Gray image - CMAP - gray')
plt.imshow(gray_img, cmap = plt.get_cmap('gray'))

plt.subplot(3,2,5)
plt.axis('off')
plt.title('Gray image - CAMP - Spring')
plt.imshow(gray_img, cmap = plt.get_cmap('spring'))

plt.subplot(3,2,6)
plt.axis('off')
plt.title('Gray image - CAMP - Ocean')
plt.imshow(gray_img, cmap = plt.get_cmap('ocean'))

plt.show()


# In[ ]:




