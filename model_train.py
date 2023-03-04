#-----------Libraries for feature extraction and training----------#
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
from feature_extraction import FlattenData


# Function train the model using grid search and dump the model as pkl file
def get_model(x_train, y_train, x_test):

    # Parameter grid to search through
    param_grid = [
        {'C':[1, 10, 100, 1000], 'kernel':['linear', 'sigmoid']},
        {'C':[1, 10, 100, 1000], 'gamma':[0.001, 0.0001, 'auto'] ,'kernel':['linear', 'rbf', 'poly']}
    ]
    
    # Call the SVM model and using grid search to find best hyperparameters
    model = svm.SVC()
    clf = GridSearchCV(model, param_grid=param_grid, verbose=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    print('>Accuracy of the model: ', accuracy_score(y_pred, y_test))
    print('>Best hyperparameters: ', clf.best_params_)
    print('>Best score: ', clf.best_score_)

    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)

if __name__ == '__main__':
    
    # Load the train test split files
    train_test_data = FlattenData(data_dir='data')
    x_train, x_test, y_train, y_test = train_test_data.flatten_data()

    # Executing the user defined function from above
    get_model(x_train, y_train, x_test)