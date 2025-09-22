from sklearn.naive_bayes import GaussianNB
import joblib
from data import load_data, split_data


def fit_model(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifer and save the trained model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    joblib.dump(nb_model, "../model/wine_model.pkl")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
    joblib.dump((X_test, y_test), "../model/test_data.pkl")





   