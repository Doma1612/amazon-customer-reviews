from src.core.data.data_loader import DataLoader

data_loader = DataLoader()

# must load final data here
customer_reviews = data_loader.get_customer_reviews("final")
train, test = data_loader.get_train_test_split(customer_reviews)
