
import numpy
from matplotlib import pyplot as plt
import argparse
import cv2
from PIL import Image
#im = Image.open("p.jpg")
im= cv2.imread('grant.jpg')
#cv2.imshow("image",im)
#cv2.waitKey(0)

grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", grey)
cv2.waitKey(0)
hist = cv2.calcHist([grey], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
