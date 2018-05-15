import tensorflow as tf

image = tf.image.decode_png(tf.read_file("p.jpg"),channels=1)

