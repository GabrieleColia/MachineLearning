import tensorflow as tf
import cv2
import re
import random
import os

def segmentation(imgPath, pathOutput):
    img = cv2.imread(imgPath)
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(imgGrey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite(pathOutput, th)

def pre_process(pathImage, training, pathOutput,numIter):

    img_size_cropped=500

    image = tf.image.decode_png(tf.read_file(pathImage), channels=3)

    session = tf.InteractiveSession()
    type(tf.constant([1,2,3]).eval())
    i=0
    while i<numIter:

        offset_height= 0
        offset_width = 0
        target_height = 600
        target_width = 600

        # Crop the image as per the parameters
        modified_image_tensor = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

        modified_image_tensor = tf.random_crop(modified_image_tensor, size=[img_size_cropped, img_size_cropped, 3])
        modified_image_tensor = tf.image.random_flip_left_right(modified_image_tensor)
        modified_image_tensor = tf.image.random_flip_up_down(modified_image_tensor)

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

        output_image = tf.image.encode_png(modified_image_tensor)

        # Create a constant as filename
        file_name2 = pathOutput.split(".")
        string=file_name2[0]+str(i)+".jpg"
        file_name = tf.constant(string)
        file = tf.write_file(file_name, output_image)

        session.run(file)
        i = i + 1
    session.close()

    return output_image

file_txt = open("elenco.txt",'r')

for line in file_txt:
    #lineSplit = riga del file elenco.txt
    lineSplit= re.split('\t',line)

    #path = percorso singola immagine
    path=lineSplit[1]

    #pathSplit = lista dei nodi path
    pathSplit = re.split(r'[ /]',path)

    pathSplit[0]="segmented"
    pathOutputSeg='/'.join(pathSplit)
    pathInputPre=pathOutputSeg
    pathDirSeg = pathOutputSeg.rsplit('/', 1)[0]

    if not os.path.exists(pathDirSeg):
        os.makedirs(pathDirSeg)

    segmentation(path, pathOutputSeg)

    pathSplit = re.split(r'[ /]', path)
    pathSplit[0] = "preprocessed"
    pathOutput = '/'.join(pathSplit)

    pathDirPre = pathOutput.rsplit('/', 1)[0]

    if not os.path.exists(pathDirPre):
        os.makedirs(pathDirPre)

    pre_process(pathInputPre,True, pathOutput, 2)
