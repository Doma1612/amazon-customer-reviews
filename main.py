from src.core.data.data_loader import DataLoader
from src.core.data.data_transformer import DataTransformer
import joblib
import pandas as pd

def main():

    # reviews_file = "data/raw/Musical_Instruments.jsonl.gz"
    # metadata_file = "data/raw/meta_Musical_Instruments.jsonl.gz"
    # output_file_csv = "data/raw/merged_reviews_metadata_renamed.csv"

    # print("Loading data...")
    # loader = DataLoader(reviews_file, metadata_file)
    # loader.load_data(sample_size=20000)
    # loader.drop_columns()

    # print(f"Reviews Shape: {loader.reviews_df.shape}")
    # print(f"Metadata Shape: {loader.metadata_df.shape}")

    # print("Transforming data...")
    # transformer = DataTransformer(loader.reviews_df, loader.metadata_df)
    # transformer.handle_missing_values()
    # transformer.remove_duplicates()
    # transformer.merge_and_rename()
    # transformer.save_merged_data_csv(output_file_csv, sep=";")

    # print("Merged DataFrame Preview:")
    # print(transformer.merged_df.head())
    # print(f"Merged DataFrame Shape: {transformer.merged_df.shape}")

    # print("Starting feature engineering...")
    # transformer.preprocess()

    # print("Processed DataFrame Preview:")
    # print(transformer.merged_df.head())
    # print(f"Processed DataFrame Shape: {transformer.merged_df.shape}")

    # print("Saving processed data and models...")
    # transformer.save_processed_data(
    #     X_path="data/processed/X_features.joblib",
    #     y_path="data/processed/y_target.joblib",
    #     tfidf_path="data/models/tfidf_vectorizer.joblib",
    #     svd_tfidf_path="data/models/svd_tfidf.joblib",
    #     svd_embeddings_path="data/models/svd_embeddings.joblib"
    # )

    # print("Data preprocessing and feature engineering completed.")

    X_path = "data/processed/X_features.joblib"
    y_path = "data/processed/y_target.joblib"
    svd_embeddings_path = "data/models/svd_embeddings.joblib"

    data_loader = DataLoader(reviews_file="", metadata_file="")

    X, y, svd_embeddings = data_loader.load_processed_data(X_path=X_path, y_path=y_path, svd_embeddings_path=svd_embeddings_path)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

if __name__ == "__main__":
    main()
