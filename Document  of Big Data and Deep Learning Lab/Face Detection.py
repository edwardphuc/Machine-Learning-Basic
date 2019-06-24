#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import urllib 
import matplotlib.pyplot as plt 
import cv2

#funcion to read image from url
def url2image(url):
    resp = urlopen(url) # download image from url
    image = np.asarray(bytearray(resp.read()),dtype ='uint8') #convert it to a Numpy array
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) # read it into OpenCV format
    return image
import sys
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen
import os.path

#Main
#load image from url
bgr_img = url2image('https://upload.wikimedia.org/wikipedia/en/thumb/7/7d/Lenna_%28test_image%29.png/220px-Lenna_%28test_image%29.png')
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
face_classifier_xml = 'my_haarcascade_frontalface_default.xml'
ret = os.path.exists(face_classifier_xml)

if ret:
    print('the cascade classifier xml file already existed\n')
else:
    print('downloading the cascade classifier xml file from internet...\n')

face_classifier_url = 'https://raw.githubusercontent.com/shantnu/Webcam-Face-Detect/master/'+'haarcascade_frontalface_default.xml'

resp = urlopen(face_classifier_url)
data=resp.read()
fh = open(face_classifier_xml, 'wb')
fh.write(data)
fh.close()
resp.close()

face_cascade = cv2.CascadeClassifier(face_classifier_url)
faces = face_cascade.detectMultiScale(gray_img, 1.25, 3)
for(x,y,w,h) in faces:
    cv2.rectangle(bgr_img,(x,y),(x+w,y+h),(255,0,0),2)
    
plt.axis('off')
plt.title('face detecion result')
plt.imshow(cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB))
plt.show()


# In[13]:


import numpy as np
import urllib 
import matplotlib.pyplot as plt 
import cv2

#funcion to read image from url
def url2image(url):
    resp = urlopen(url) # download image from url
    image = np.asarray(bytearray(resp.read()),dtype ='uint8') #convert it to a Numpy array
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) # read it into OpenCV format
    return image
import sys
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen
import os.path

#Main
#load image from url
bgr_img = url2image('https://phunghuy.files.wordpress.com/2012/10/steve-jobs1.jpg?w=1140')
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
face_classifier_xml = 'my_haarcascade_frontalface_default.xml'
ret = os.path.exists(face_classifier_xml)

if ret:
    print('the cascade classifier xml file already existed\n')
else:
    print('downloading the cascade classifier xml file from internet...\n')

face_classifier_url = 'https://raw.githubusercontent.com/shantnu/Webcam-Face-Detect/master/'+'haarcascade_frontalface_default.xml'

resp = urlopen(face_classifier_url)
data=resp.read()
fh = open(face_classifier_xml, 'wb')

fh.write(data)
fh.close()
resp.close()

face_cascade = cv2.CascadeClassifier(face_classifier_xml)
faces = face_cascade.detectMultiScale(gray_img, 1.25, 3)
for(x,y,w,h) in faces:
    cv2.rectangle(bgr_img,(x,y),(x+w,y+h),(255,0,0),2)
    
plt.axis('off')
plt.title('face detecion result')
plt.imshow(cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB))
plt.show()


# In[ ]:




