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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm

#-----------Load the normalized data----------#
""" image_batch = np.load('train_image_batch_data.npy')
labels_batch = np.load('train_labels_batch_data.npy')
val_data = tf.data.experimental.load('val_data') """
with open('train_test_files.pkl', 'rb') as f:
    x_train, x_test, y_train, y_test = pickle.load(f)


if __name__ == '__main__':
    # Data properties
    """ image_height = image_batch.shape[1]
    image_width = image_batch.shape[2]

    #-----------Build the Model----------#

    # Let's build a Sequential model based on CNN 
    model = Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Show the summary of the model
    model.summary()

    # Train the model
    model.fit(image_batch, labels_batch, validation_data=val_data, batch_size=64, epochs=50)

    if not os.path.exists('cnn_model.pkl'):
        with open('/cnn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        print('File already exists') """

    param_grid = [
        {'C':[1, 10, 100, 1000], 'kernel':['linear']},
        {'C':[1, 10, 100, 1000], 'gamma':[0.001, 0.0001] ,'kernel':['rbf']}
    ]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid=param_grid, verbose=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print('Accuracy of the model : ', accuracy_score(y_pred, y_test))

    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)