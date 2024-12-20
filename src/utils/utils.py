import pandas as pd

data_path = "data/DATA_STATE/amazon_customer_reviews.py"
placeholder = "PLACEHOLDER"
def get_dataframe(data_state: str):
    path = data_path.replace(placeholder, data_state)
    df = pd.read_csv(path)
    return df