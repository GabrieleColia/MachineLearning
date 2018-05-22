from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

#hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))

#output layer  modificare units
classifier.add(Dense(units = 3, activation = 'sigmoid'))

#compila il classificatore
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#preprocessing per evitare overfitting
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/images/lab',target_size = (64, 64),batch_size = 32,class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('test',target_size = (64, 64),batch_size = 32,class_mode = 'categorical')

classifier.fit_generator(training_set,steps_per_epoch = 281,epochs = 10,validation_data = test_set,validation_steps = 100)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('alba.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 0:
    prediction = 'albatross'
elif result[0][0] == 1:
    prediction = 'Fulmar'
else:
    prediction = 'pellicano'

print(prediction)