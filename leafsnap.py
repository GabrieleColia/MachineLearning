import imgaug

import numpy
from matplotlib import pyplot as plt
import argparse
import cv2
from PIL import Image

import tensorflow as tf

im= cv2.imread('grant.jpg')

def pre_process_image(image,training):
    image2 = tf.image.decode_png(tf.read_file("p.jpg"), channels=1)
    if training:
        croppedimage2 = tf.random_crop(image, size=[10, 10, 1])
        #image2 = tf.minimum(image, 1.0)
        #image2 = tf.maximum(image, 0.0)
    return tf.image.encode_png(croppedimage2)

im2=pre_process_image(im,True)
file_name = tf.constant('./data/im2.png')
file = tf.write_file(file_name, im2)

#cv2.imshow("Im", im2)
#cv2.waitKey(0)

