import csv
import cv2
import numpy as np
import tensorflow as tf
import os
import keras
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(tf.__version__)
# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


def normalized_image(x):
    return (x/255-0.5)


model = Sequential()
# Normalize the model
model.add(Lambda(normalized_image, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 20), (0, 0))))
# model.output_shape == (None, 90, 320, 3)

# model.add(Lambda(normalized_image, input_shape=(160, 320, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
callback = keras.callbacks.TensorBoard(log_dir='./logs')
model.fit(X_train, y_train, validation_split=0.2,
          shuffle=True, batch_size=32, epochs=30, callbacks=callback)
model.save('model.h5')
