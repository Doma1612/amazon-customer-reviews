from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from typing import Tuple

from utils.config import X_features, y_target

# Encoding the labels to ensure they're numerically from 0 to num_classes-1
label_encoder = LabelEncoder()
y_target_encoded = label_encoder.fit_transform(y_target)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target_encoded, test_size=0.2, random_state=42, stratify=y_target_encoded
)

# Convert sparse matrices to dense
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Define the neural network
class CustomerReviewNN(nn.Module):
    def __init__(self, input_size=300, hidden_size=100, num_classes=5):
        super(CustomerReviewNN, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)
    
    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

def get_loss_fn_and_optimizer(model: CustomerReviewNN, learning_rate: float, device="cpu") -> Tuple:
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Anpassen der Verlustfunktion
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer

# Train function
def train(model, optimizer, loss_fn, X_train, y_train, num_epochs=1, device="cpu"):
    model.train()
    for epoch in range(num_epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
        labels = torch.tensor(y_train, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def get_evaluation_metrics(y_test, y_pred):
    test_accuracy = accuracy_score(y_test, y_pred)

    print("Classification Report:")
    evaluation_report = classification_report(y_test, y_pred)
    print(evaluation_report)

    print(f"Test Accuracy: {test_accuracy:.2f}")
    return test_accuracy, evaluation_report

def display_confusion_matrix(y_test, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.show()

def test(model, X_test, y_test, device="cpu"):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        labels = torch.tensor(y_test, dtype=torch.long).to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Evaluate metrics using utility functions
        get_evaluation_metrics(labels.cpu(), predicted.cpu())
        display_confusion_matrix(labels.cpu(), predicted.cpu())

def main():
    # Parameters
    learning_rate = 0.01
    num_epochs = 50
    hidden_size = 100

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    input_size = X_train.shape[1]
    num_classes = len(set(y_target_encoded))  # Number of unique classes
    model = CustomerReviewNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)

    # Get loss function and optimizer
    loss_fn, optimizer = get_loss_fn_and_optimizer(model, learning_rate)

    # Train the model
    trained_model = train(model, optimizer, loss_fn, X_train_dense, y_train, num_epochs=num_epochs, device=device)

    # Test the model
    test(trained_model, X_test_dense, y_test, device=device)

if __name__ == "__main__":
    main()