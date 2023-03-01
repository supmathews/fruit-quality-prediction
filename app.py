#-----------Libraries for feature extraction----------#
import os
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
from skimage.transform import resize
import model_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

CATEGORIES = ['no_split', 'split']

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    #file = request.files['image']
    url = request.form['url']
    #filename = secure_filename(file.filename)
    #img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #file.save(img_path)
    #print(f'\nFile saved successfully at {img_path}\n')
    flat_data = model_prediction.extract_features(url)
    result = model.predict(flat_data)
    result = CATEGORIES[result[0]]
    return result


if __name__ == '__main__':
    app.run(debug=True)