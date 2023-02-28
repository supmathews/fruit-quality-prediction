#-----------Libraries for feature extraction----------#
import os
import numpy as np
from PIL import Image
import pickle
import pathlib
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras import layers

#-----------Load the images----------#
data_dir = 'data'
batch_size = 64
img_height = 224
img_width = 224

# split the data for training and validation using keras's preprocessing
train_data = preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset ='training',
    seed =123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

val_data = preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset ='validation',
    seed =123,
    image_size = (img_height, img_width),
    batch_size = batch_size
) 
print('\nClasses: ', train_data.class_names)
# Cache the images
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(3500).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

print('\nTrain Data: \n', train_data)

# Normalizing the data
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
print('Normalized Training Data:\n')
print(normalized_train_data)

image_batch, labels_batch = next(iter(normalized_train_data))
print('Image batch :\n', image_batch)
print('Labels batch :\n', labels_batch)

#-----------Store the data----------#
# Store the normalized data as np file
np.save('train_image_batch_data.npy', image_batch)
np.save('train_labels_batch_data.npy', labels_batch)
#np.save('val_data.npy', val_data)