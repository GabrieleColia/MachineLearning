import tensorflow as tf
import cv2



def pre_process(pathImage, training):

    img_size_cropped=500

    image = tf.image.decode_png(tf.read_file(pathImage), channels=1)
    im = cv2.imread(pathImage)
    session = tf.InteractiveSession()

    if(training):


        # Import the image


        ## Set the variable values here
        # Offset variables values
        offset_height= 20
        offset_width = 20
        # Target variables values
        target_height = 20
        target_width = 20

        # Begin the session


        #print(session.run(image))

        # Crop the image as per the parameters
        modified_image_tensor = tf.image.random_saturation(im, lower=0.0, upper=2.0)
        modified_image_tensor = tf.image.random_hue(modified_image_tensor, max_delta=0.25)
        modified_image_tensor = tf.random_crop(modified_image_tensor, size=[img_size_cropped, img_size_cropped, 3])
        modified_image_tensor = tf.image.random_flip_left_right(modified_image_tensor)
        modified_image_tensor = tf.image.random_flip_up_down(modified_image_tensor)

        modified_image_tensor = tf.image.random_contrast(modified_image_tensor, lower=0.3, upper=1.0)
        modified_image_tensor = tf.image.random_brightness(modified_image_tensor, max_delta=0.2)


        #modified_image_tensor = tf.minimum(modified_image_tensor, 1.0)
        #modified_image_tensor = tf.maximum(modified_image_tensor, 0.0)

    else:
        modified_image_tensor = tf.image.resize_image_with_crop_or_pad(image, target_height=img_size_cropped, target_width=img_size_cropped)

    output_image = tf.image.encode_png(modified_image_tensor)

    # Create a constant as filename
    file_name = tf.constant('./output/Ouput_image' + str(a) + '.png')
    file = tf.write_file(file_name, output_image)

    print(session.run(file))

    print("Image Saved!")

    session.close()

    return output_image

a=30
while(a!=0):
    pre_process("a.png",True)
    a=a-1