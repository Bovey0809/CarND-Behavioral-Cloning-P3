import csv
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
csv_file = './data/driving_log.csv'


def process_image(x):
    return x


# car_images = []
# steering_angles = []
# with open(csv_file, 'r') as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         steering_center = float(row[3])

#         # create adjusted steering measurements for the side camera images
#         correction = 0.2  # this is a parameter to tune
#         steering_left = steering_center + correction
#         steering_right = steering_center - correction

#         # read in images from center, left and right cameras
#         path = "./data/IMG/"  # fill in the path to your training IMG directory
#         img_center = process_image(np.asarray(
#             Image.open(path + row[0].split('/')[-1])))
#         img_left = process_image(np.asarray(
#             Image.open(path + row[1].split('/')[-1])))
#         img_right = process_image(np.asarray(
#             Image.open(path + row[2].split('/')[-1])))

#         # add images and angles to data set
#         car_images.extend([img_center, img_left, img_right])
#         steering_angles.extend(
#             [steering_center, steering_left, steering_right])

lines = []
with open('./data/driving_log.csv') as csvfile:
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
