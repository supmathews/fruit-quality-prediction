#-----------Libraries for feature extraction----------#
import os
import numpy as np
from PIL import Image
import pickle
import pathlib
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from keras.models import Sequential

#-----------Load the normalized data----------#
image_batch = np.load('train_image_batch_data.npy')
labels_batch = np.load('train_labels_batch_data.npy')
#val_data = np.load('val_data.npy', allow_pickle=True)

# Data properties
image_height = image_batch.shape[1]
image_width = image_batch.shape[2]

#-----------Split the ----------#


#-----------Build the Model----------#

# Let's build a Sequential model based on CNN 
model = Sequential()
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
model.add(layers.MaxPool2D(2))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(image_batch, labels_batch, batch_size=64, epochs=50)
print(history)