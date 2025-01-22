# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC

# Load numpy
import numpy as np

from utils.utils import get_evaluation_metrics, display_confusion_matrix
from utils.config import X_features, y_target

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42, stratify=y_target)

def log_reg_classification():
    print("Starting logistic regression")

    # Initialisiere das Logistische Regressionsmodell
    logreg = LogisticRegression(random_state=42)

    # Trainiere das Modell
    logreg.fit(X_train, y_train)

    # Mache Vorhersagen auf dem Testset
    y_pred = logreg.predict(X_test)

    test_accuracy, evaluation_report = get_evaluation_metrics(y_test, y_pred)
    print(test_accuracy)
    print(evaluation_report)

    display_confusion_matrix(y_test, y_pred)
    print("Finished logistic regression")

    return test_accuracy, evaluation_report

def svm_classification():
    print("Start Support Vector Machine")
    svm = SVC(random_state=42)  # verbose=True zeigt mehr Details über den Fortschritt an

    # Trainiere das Modell
    svm.fit(X_train, y_train)

    # Mache Vorhersagen auf dem Testset
    y_pred = svm.predict(X_test)

    test_accuracy, evaluation_report = get_evaluation_metrics(y_test, y_pred)

    display_confusion_matrix(y_test, y_pred)
    print("Finished Support Vector Machine")

    return test_accuracy, evaluation_report

def rf_classification():
    print("Start random forest classifier")
    rf = RandomForestClassifier(random_state=42)

    # Trainiere das Modell
    rf.fit(X_train, y_train)

    # Mache Vorhersagen auf dem Testset
    y_pred = rf.predict(X_test)

    test_accuracy, evaluation_report = get_evaluation_metrics(y_test, y_pred)

    display_confusion_matrix(y_test, y_pred)
    print("Finished random forest classifier")

    return test_accuracy, evaluation_report

def main():
    log_reg_classification()
    rf_classification()
    svm_classification()