import tensorflow as tf
#import cv2
import re
import random

def pre_process(pathImage, training, pathOutput,numIter):

    img_size_cropped=500

    image = tf.image.decode_png(tf.read_file(pathImage), channels=3)
    #im = cv2.imread(pathImage)
    session = tf.InteractiveSession()
    i=0
    while i<numIter:
        #if(training):


        # Import the image


        ## Set the variable values here
        # Offset variables values
        offset_height= 0
        offset_width = 0
        # Target variables values
        target_height = 600
        target_width = 600

        # Begin the session


        #print(session.run(image))

        # Crop the image as per the parameters
        modified_image_tensor = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
        modified_image_tensor = tf.image.random_saturation(modified_image_tensor, lower=0.0, upper=2.0)
        modified_image_tensor = tf.image.random_hue(modified_image_tensor, max_delta=0.25)
        modified_image_tensor = tf.random_crop(modified_image_tensor, size=[img_size_cropped, img_size_cropped, 3])
        modified_image_tensor = tf.image.random_flip_left_right(modified_image_tensor)
        modified_image_tensor = tf.image.random_flip_up_down(modified_image_tensor)

        modified_image_tensor = tf.image.random_contrast(modified_image_tensor, lower=0.3, upper=1.0)
        modified_image_tensor = tf.image.random_brightness(modified_image_tensor, max_delta=0.2)

        rand = random.randint(1, 4)
        if rand==1:
            modified_image_tensor = tf.image.rot90(modified_image_tensor)
        elif rand==2:
            modified_image_tensor = tf.image.rot90(modified_image_tensor)
            modified_image_tensor = tf.image.rot90(modified_image_tensor)
        elif rand==3:
            modified_image_tensor = tf.image.rot90(modified_image_tensor)
            modified_image_tensor = tf.image.rot90(modified_image_tensor)
            modified_image_tensor = tf.image.rot90(modified_image_tensor)

        #modified_image_tensor = tf.minimum(modified_image_tensor, 1.0)
        #modified_image_tensor = tf.maximum(modified_image_tensor, 0.0)

        #else:
         #   modified_image_tensor = tf.image.resize_image_with_crop_or_pad(im, target_height=img_size_cropped, target_width=img_size_cropped)

        output_image = tf.image.encode_png(modified_image_tensor)

        # Create a constant as filename

        file_name2 = pathOutput.split(".")
        # print((listPath))
        string=file_name2[0]+str(i)+".jpg"
        file_name = tf.constant(string)
        file = tf.write_file(file_name, output_image)

        session.run(file)
        i = i + 1
    session.close()

    return output_image

file_txt = open("elenco.txt",'r')
for line in file_txt:
    lineSplit= re.split('\t',line)
    #listPathImages.append(listSplit[1])


    #a=0
    #while a < len(list):
    im=lineSplit[1]
    imagePath = re.split(r'[ /]',im)
    #print((listPath))
    imagePath[0]="preprocessing"
    pathOutput='/'.join(imagePath)
    pre_process(im,True, pathOutput, 1)
