import tensorflow
import numpy
from matplotlib import pyplot as plt
import argparse
import cv2

from PIL import Image

im= cv2.imread('p.jpg')

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
#plt.show()

chans = cv2.split(im)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []

# loop over the image channels
for (chan, color) in zip(chans, colors):
    # create a histogram for the current channel and
    # concatenate the resulting histograms for each
    # channel
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)

    # plot the histogram
    plt.plot(hist, color=color )
    plt.xlim([0, 256])

# let's move on to 2D histograms -- I am reducing the#
# number of bins in the histogram from 256 to 32 so we
# can better visualize the results
fig = plt.figure()

# plot a 2D color histogram for green and blue
#ax = fig.add_subplot(131)
hist = cv2.calcHist([im], [0, 1, 2], None,[10, 10, 10], [0, 256, 0, 256, 0, 256])
#p = ax.imshow(hist, interpolation="nearest")
#ax.set_title("2D Color Histogram for Green and Blue")
#plt.colorbar(p)


#plt.show()

print( "3D histogram shape:",hist.shape, " with", hist.flatten().shape[0], "values")