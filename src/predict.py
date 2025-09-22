import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load("../model/wine_model.pkl")
    y_pred = model.predict(X)
    return y_pred

def get_metrics():
    """
    Evaluate the trained Wine model on the test dataset.

    Loads the test data, predicts class labels using the trained model
    and calculates evaluation metrics: accuracy, precision, and recall.

    Returns:
        acc (float): Accuracy of the model on the test set rounded to 5 decimal place.
        precision (float): Precision rounded to 5 decimal place.
        recall (float): Recall rounded to 5 decimal place.
    """
    X_test, y_test = joblib.load("../model/test_data.pkl")
    y_pred = predict_data(X_test)
    acc = round(accuracy_score(y_test, y_pred),5) 
    precision = round(precision_score(y_test, y_pred, average="macro"),5)
    recall = round(recall_score(y_test, y_pred, average="macro"),5)
    return acc, precision, recall



