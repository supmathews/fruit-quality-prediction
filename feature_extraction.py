#-----------Libraries for feature extraction----------#
import os
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

target = []
images = []
flat_data = []

#-----------Load the images----------#
data_dir = 'data'
batch_size = 64
img_height = 224
img_width = 224

CATEGORIES = ['no_split', 'split']

for category in CATEGORIES:
    class_num = CATEGORIES.index(category)
    path = os.path.join(data_dir, category)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (img_height, img_width, 3))
        flat_data.append(img_resized.flatten())
        images.append(img_resized)
        target.append(class_num)

flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)

# Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.3, random_state=1234)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

train_test_files = [x_train, x_test, y_train, y_test]

# Save the files
with open('train_test_files.pkl', 'wb') as f:
    pickle.dump(train_test_files, f)