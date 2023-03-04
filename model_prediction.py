#-----------Libraries----------#
import pickle
from feature_extraction import FlattenData

CATEGORIES = ['no_split', 'split']

if __name__ == '__main__':

    # Retrieve the flattened data
    data = FlattenData()
    flat_data = data.flatten_image()
        
    # Call the SVM model
    model = pickle.load(open('model.pkl', 'rb'))

    # Make the prediction
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    print(f'>Predicted output : {y_out}')