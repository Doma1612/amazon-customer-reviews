# # Load scikit's random forest classifier library
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split 
# from sklearn.svm import SVC

# # Load numpy
# import numpy as np

# from utils.utils import get_evaluation_metrics, display_confusion_matrix
# from utils.config import X_features, y_target

# X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42, stratify=y_target)

# def log_reg_classification():
#     print("Starting logistic regression")

#     # Initialisiere das Logistische Regressionsmodell
#     logreg = LogisticRegression(random_state=42)

#     # Trainiere das Modell
#     logreg.fit(X_train, y_train)

#     # Mache Vorhersagen auf dem Testset
#     y_pred = logreg.predict(X_test)

#     test_accuracy, evaluation_report = get_evaluation_metrics(y_test, y_pred)
#     print(test_accuracy)
#     print(evaluation_report)

#     display_confusion_matrix(y_test, y_pred)
#     print("Finished logistic regression")

#     return test_accuracy, evaluation_report

# def svm_classification():
#     print("Start Support Vector Machine")
#     svm = SVC(random_state=42)  # verbose=True zeigt mehr Details über den Fortschritt an

#     # Trainiere das Modell
#     svm.fit(X_train, y_train)

#     # Mache Vorhersagen auf dem Testset
#     y_pred = svm.predict(X_test)

#     test_accuracy, evaluation_report = get_evaluation_metrics(y_test, y_pred)

#     display_confusion_matrix(y_test, y_pred)
#     print("Finished Support Vector Machine")

#     return test_accuracy, evaluation_report

# def rf_classification():
#     print("Start random forest classifier")
#     rf = RandomForestClassifier(random_state=42)

#     # Trainiere das Modell
#     rf.fit(X_train, y_train)

#     # Mache Vorhersagen auf dem Testset
#     y_pred = rf.predict(X_test)

#     test_accuracy, evaluation_report = get_evaluation_metrics(y_test, y_pred)

#     display_confusion_matrix(y_test, y_pred)
#     print("Finished random forest classifier")

#     return test_accuracy, evaluation_report

# def main():
#     log_reg_classification()
#     rf_classification()
#     svm_classification()


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
import numpy as np
from utils.utils import get_evaluation_metrics, display_confusion_matrix
from utils.config import X_features, y_target

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42, stratify=y_target)

def log_reg_classification():
    print("Starting logistic regression")

    # Define the parameter grid for Logistic Regression
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    # Initialize the Logistic Regression model
    logreg = LogisticRegression(random_state=42, max_iter=1000)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
    
    # Train the model with GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_logreg = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_logreg.predict(X_test)

    test_accuracy, evaluation_report = get_evaluation_metrics(y_test, y_pred)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {test_accuracy}")
    print(evaluation_report)

    display_confusion_matrix(y_test, y_pred)
    print("Finished logistic regression")

    return test_accuracy, evaluation_report

def svm_classification():
    print("Start Support Vector Machine")

    # Define the parameter grid for SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    # Initialize the SVM model
    svm = SVC(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
    
    # Train the model with GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_svm = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_svm.predict(X_test)

    test_accuracy, evaluation_report = get_evaluation_metrics(y_test, y_pred)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {test_accuracy}")
    print(evaluation_report)

    display_confusion_matrix(y_test, y_pred)
    print("Finished Support Vector Machine")

    return test_accuracy, evaluation_report

def rf_classification():
    print("Start random forest classifier")

    # Define the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', verbose=1, random_state=42)
    
    # Train the model with RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get the best model
    best_rf = random_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_rf.predict(X_test)

    test_accuracy, evaluation_report = get_evaluation_metrics(y_test, y_pred)
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Test Accuracy: {test_accuracy}")
    print(evaluation_report)

    display_confusion_matrix(y_test, y_pred)
    print("Finished random forest classifier")

    return test_accuracy, evaluation_report

def main():
    #log_reg_classification()
    rf_classification()
    svm_classification()

if __name__ == "__main__":
    main()