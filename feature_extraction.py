#-----------Libraries for feature extraction----------#
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

'''
This user defined class contains method flatten_data which allows 
the user to flatten the data in order to get the pixel values of the images
in the form of arrays
'''

class FlattenData:

    def __init__(self, data_dir, batch_size=64, img_height=224, img_width=224):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width


    # Function to flatten the data
    def flatten_data(self):
        
        target = []
        flat_data = []

        CATEGORIES = ['no_split', 'split']
        
        for category in CATEGORIES:
            class_num = CATEGORIES.index(category)
            path = os.path.join(self.data_dir, category)
            
            for img in os.listdir(path):
                img_array = imread(os.path.join(path, img))
                img_resized = resize(img_array, (self.img_height, self.img_width, 3))
                flat_data.append(img_resized.flatten())
                target.append(class_num)
        
        # Convert the data to np.array format
        flat_data = np.array(flat_data)
        target = np.array(target)

        # Split the data into training and testing data
        x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.3, random_state=1234)

        # Retrun the splits
        return x_train, x_test, y_train, y_test