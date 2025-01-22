import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

data_path = "data/DATA_STATE/amazon_customer_reviews.py"
placeholder = "PLACEHOLDER"
target_column = "reviews"

def get_dataframe(data_state: str):
    path = data_path.replace(placeholder, data_state)
    df = pd.read_csv(path)
    return df

def get_evaluation_metrics(y_test, y_pred):
    test_accuracy = accuracy_score(y_test, y_pred)

    # Zeige die Ergebnisse
    print("Classification Report:")
    evaluation_report = classification_report(y_test, y_pred)
    print(evaluation_report)

    print(f"Test Accuracy: {test_accuracy:.2f}")
    return test_accuracy, evaluation_report

def display_confusion_matrix(y_test, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.show()
