# coding: utf-8
import csv
import cv2
import numpy as np
import tensorflow as tf
import os
import keras
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('GPU', '0', 'The gpu you want to use.')
tf.app.flags.DEFINE_integer('epochs', 10, 'Epochs')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_string('log_dir', './logs/', 'log directory.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.app.flags.DEFINE_string('save_dir', './models/',
                           'directory for saving model')
tf.app.flags.DEFINE_string(
    'data_dir', './IMG', 'dataset address and csv file address.')
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU

# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
lines = []
with open(FLAGS.data_dir+'/driving_log_new.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        steering_center = float(line[3])
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        img_center = line[0].split('/')[-1]
        img_left = line[1].split('/')[-1]
        img_right = line[2].split('/')[-1]

        lines.append((img_center, steering_center))
        lines.append((img_right, steering_right))
        lines.append((img_left, steering_left))

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    batch_size = batch_size // 2
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = FLAGS.data_dir+'/IMG/'+batch_sample[0]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)
                # image augment by flipping.
                image_flipped = np.fliplr(center_image)
                measurement_flipped = -1. * center_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)

            # trim image to only see section with road
            X_train = np.array(images)

            # print(len(images), X_train.shape)

            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# test generator
# _test_generator = generator(train_samples)
# _test_data = next(_test_generator)
# images = _test_data[0]
# print('the original shape of image {}'.format(images[0].shape))
# angles = _test_data[1]
# index = np.random.randint(16)
# plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
# plt.title("drive {}, shape{}".format(angles[index], images[index].shape))


# Parameters for generator
train_size = len(train_samples)
valid_size = len(validation_samples)
steps_per_epoch = train_size // FLAGS.batch_size
validation_steps = valid_size // FLAGS.batch_size
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
validation_generator = generator(
    validation_samples, batch_size=FLAGS.batch_size)


ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
                 input_shape=(row, col, ch)))
model.add(Cropping2D(((70, 25), (0, 0))))
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# callbacks
callback = keras.callbacks.TensorBoard(
    log_dir=FLAGS.log_dir+"{}-{}-{}".format(FLAGS.epochs, FLAGS.learning_rate, FLAGS.batch_size))
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=32, epochs=30, callbacks=callback)
# for generator
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=steps_per_epoch, epochs=FLAGS.epochs, validation_steps=validation_steps,
                    callbacks=[callback, early_stopping], use_multiprocessing=True, workers=6)

# Save model and weights
if not os.path.isdir(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

model.save(FLAGS.save_dir+'{}-{}-{}.h5'.format(FLAGS.epochs,
                                               FLAGS.learning_rate, FLAGS.batch_size))

# plt.plot(model.history.history['loss'])
# model.summary()
