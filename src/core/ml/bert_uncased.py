import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --------------------
# DATASET DEFINITION
# --------------------
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]  # 1-5
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        # Anpassen: 1-5 auf 0-4
        item['labels'] = torch.tensor(label - 1)
        return item

# --------------------
# FUNCTIONS
# --------------------
def load_data(csv_path='./data/merged_reviews_metadata_renamed.csv', sample_size=10000):
    print("Lade Daten...")
    df = pd.read_csv(csv_path, sep=';', nrows=sample_size)
    df['combined_text'] = df['review_title'] + " " + df['review_text']
    df = df[['review_rating', 'combined_text']]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.dropna(subset=['combined_text'])
    test_df = test_df.dropna(subset=['combined_text'])
    return train_df, test_df

def prepare_dataloaders(train_df, test_df, tokenizer, batch_size=16):
    print("Erstelle DataLoader...")
    train_dataset = ReviewDataset(train_df['combined_text'], train_df['review_rating'], tokenizer)
    test_dataset = ReviewDataset(test_df['combined_text'], test_df['review_rating'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_model(model, train_loader, device, epochs=3, lr=2e-5, weight_decay=0.01):
    print("Starte Training...")
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = len(train_loader)
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        for i, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 0 or i == num_batches:
                avg_loss = running_loss / i
                print(f"Batch {i}/{num_batches} - Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1} abgeschlossen (durchschnittliche Loss: {avg_loss:.4f})")

def evaluate_model(model, test_loader, device):
    print("\nStarte Evaluation...")
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch['labels'].cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[f"Klasse {i}" for i in range(5)])
    print(f"\nAccuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

def save_model(model, path_weights="./data/models/model_weights.pth", path_full="./data/models/model_full.pth"):
    print("Speichere Modell...")
    torch.save(model.state_dict(), path_weights)
    torch.save(model, path_full)
    print("Modelle gespeichert.")

def predict_review(review_text, tokenizer, model, device):
    encoding = tokenizer(review_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
    pred = torch.argmax(outputs.logits, dim=1)
    return pred.item() + 1


def main():
    train_df, test_df = load_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    train_loader, test_loader = prepare_dataloaders(train_df, test_df, tokenizer)
    device = get_device()
    print("Device:", device)

    train_model(model, train_loader, device, epochs=3, lr=2e-5, weight_decay=0.01)
    save_model(model)

    evaluate_model(model, test_loader, device)

    text = "the product was great"
    result = predict_review(text, tokenizer, model, device)
    print(f"Vorhergesagte Sternebewertung für Beispiel: {result}")

if __name__ == "__main__":
    main()
