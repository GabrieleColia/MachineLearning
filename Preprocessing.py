import tensorflow as tf
a=10
while a!=0:

    # Import the image
    image = tf.image.decode_png(tf.read_file("p.jpg"), channels=1)

    ## Set the variable values here
    # Offset variables values
    offset_height= 20
    offset_width = 20
    # Target variables values
    target_height = 20
    target_width = 20

    # Begin the session
    session = tf.InteractiveSession()

    print(session.run(image))

    # Crop the image as per the parameters
    cropped_image_tensor = tf.random_crop(image, size=[100, 100, 1])

    output_image = tf.image.encode_png(cropped_image_tensor)

    # Create a constant as filename
    file_name = tf.constant('./data/Ouput_image'+str(a)+'.png')
    file = tf.write_file(file_name, output_image)

    print(session.run(file))

    print("Image Saved!")

    session.close()

    a=a-1

