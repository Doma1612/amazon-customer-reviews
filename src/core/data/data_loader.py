import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.utils import get_dataframe

#  To-Do: Implement data loader

class DataLoader():
    def __init__():
        pass

    def get_customer_reviews(data_state: str):
        """Get customer review dataframe in specified state

        Args:
            data_state (str): String that determines the foldername of the data

        Returns:
            pd.DataFrame: Customer review dataframe.
        """
        df = get_dataframe(data_state=data_state)
        return df

    def get_train_test_split(df, y):
        """Return a train test split with defined settings for any dataset.

        Args:
            df (DataFrame): Dataframe.
            y (str): Name of the target column.

        Returns:
            tuple: Train and Test df.
        """
        train, test = train_test_split(df, test_size=0.2, stratify=df[y], random_state=42)
        return train, test
    
# index label mapping in preprocessing? final data should contain that column
