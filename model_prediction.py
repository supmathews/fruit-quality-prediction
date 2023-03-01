#-----------Libraries for feature extraction----------#
import os
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize

CATEGORIES = ['no_split', 'split']

if __name__ == '__main__':

    flat_data = []
    url = input('Enter image url : ')
    img = imread(url)
    img_resized = resize(img, (224, 224, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    
    print(img.shape)
    
    plt.imshow(img_resized)
    
    # Call the CNN model
    model = pickle.load(open('model.pkl', 'rb'))

    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    print(f'>Predicted output : {y_out}')