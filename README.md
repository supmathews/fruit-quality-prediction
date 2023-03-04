# CLI Application to Predict Good and Bad Tomatoes
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)                 
[![Python 3.8.5](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/) 
### This repo looks into classifying images of tomatoes as good or bad(split or torn) using machine learning.

## Features
- This is an image classification CLI application making use of [SVM](https://scikit-learn.org/stable/modules/svm.html) model to predict.
- The dataset was flattened using [Scikit-image](https://scikit-image.org/) library.

## Getting Started
- Clone this repository.
- Open CMD or terminal in working directory.
- Run following command to download dependencies.
  ```
  pip install -r requirements.txt
  ```
- Run following command to test the application
  ```
  python model_prediction.py
  ```
- Enter the url of the image in the terminal. The url can be a local path or a web address.
![Screenshot](screenshot\how-to.jpg)


