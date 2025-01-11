import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from scipy import sparse
from sentence_transformers import SentenceTransformer

from src.utils.utils import get_dataframe

#  To-Do: Implement data loader


class DataLoader:
    def __init__(self, reviews_file: str, metadata_file: str):
        self.reviews_file = reviews_file
        self.metadata_file = metadata_file
        self.reviews_df = None
        self.metadata_df = None

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
        train, test = train_test_split(
            df, test_size=0.2, stratify=df[y], random_state=42
        )
        return train, test

    def load_data(self, sample_size: int = 20000) -> None:
        self.reviews_df = pd.read_json(
            self.reviews_file, lines=True, compression="gzip"
        ).sample(n=sample_size, random_state=42)
        self.metadata_df = pd.read_json(
            self.metadata_file, lines=True, compression="gzip"
        )

    def drop_columns(self) -> None:
        self.metadata_df = self.metadata_df[
            [
                "main_category",
                "title",
                "average_rating",
                "rating_number",
                "store",
                "parent_asin",
            ]
        ]
        self.reviews_df = self.reviews_df[
            [
                "rating",
                "title",
                "text",
                "asin",
                "parent_asin",
                "user_id",
                "verified_purchase",
            ]
        ]

    def load_processed_data(
        self,
        X_path="X_features_reduced.joblib",
        y_path="y_target_reduced.joblib",
        svd_embeddings_path="svd_embeddings_reduced.joblib",
    ):
        """
        Load Feature, Target and SVD Model

        Parameters:
        - X_path (str): Path to Feature-Matrix.
        - y_path (str): Path to Target.
        - svd_embeddings_path (str): Path to Embedding SVD Modell.

        Returns:
        - X (scipy.sparse.csr_matrix): Feature-Matrix.
        - y (numpy.ndarray): Target.
        - svd_embeddings (TruncatedSVD): SVD Model for Embeddings.
        """

        X = joblib.load(X_path)

        y = joblib.load(y_path)

        svd_embeddings = joblib.load(svd_embeddings_path)

        print("Alle verarbeiteten Daten und Modelle wurden erfolgreich geladen.")

        return X, y, svd_embeddings


# index label mapping in preprocessing? final data should contain that column
