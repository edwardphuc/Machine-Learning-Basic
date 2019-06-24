# Machine Learning Basic
**Application development with smart camera**

####1.load image:
A function to read image from an url
``` python
 def url2image(url):
     resp = urlopen(url)
     image = np asarray(bytearray(resp.read()), dtype = 'uint8')
     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
```
![Hình 1](https://i.imgur.com/LeADMo4.png) \n

####2.Face Detection:
using [Haar Cascades](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html)
We have 2 step for it:
1. training system face detection
2. predict the results based on dataset
![Hình 2](https://imgur.com/YttbTjr.png)
