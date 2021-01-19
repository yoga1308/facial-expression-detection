import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = 'C:/Users/Yoga/Desktop/PY/Fer2013/train'
test_dir = 'C:/Users/Yoga/Desktop/PY/Fer2013/test'

trainDatagen = ImageDataGenerator(rescale=1/255)
testDatagen = ImageDataGenerator(rescale=1/255)

train_gen = trainDatagen.flow_from_directory(train_dir,target_size=(48,48), class_mode='categorical',batch_size=64,color_mode="grayscale")
test_gen = testDatagen.flow_from_directory(test_dir,target_size=(48,48),class_mode='categorical',batch_size=64,color_mode="grayscale")

model = tf.keras.models.Sequential([
    tf.keras.layers.Convolution2D(32,(3,3),input_shape=(48,48,1),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Convolution2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Convolution2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Convolution2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Dense(7,activation='softmax')
])

model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
model.fit_generator(train_gen,steps_per_epoch=256 ,epochs=100,validation_data=test_gen,validation_steps=64,verbose=1)
model.save('C:/Users/Yoga/Desktop/PY/Models/FaceExpressionIdentificaionModel.h5')