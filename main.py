from src.core.data.data_loader import DataLoader
from src.core.data.data_transformer import DataTransformer
import joblib
import pandas as pd


def main():
    reviews_file = "data/raw/Musical_Instruments.jsonl.gz"
    metadata_file = "data/raw/meta_Musical_Instruments.jsonl.gz"
    output_file_csv = "data/raw/merged_reviews_metadata_renamed.csv"

    print("Loading data...")
    loader = DataLoader(reviews_file, metadata_file)
    loader.load_data(sample_size=20000)
    loader.drop_columns()

    print("Transforming data...")
    transformer = DataTransformer(loader.reviews_df, loader.metadata_df)
    
    # Calculate distrubution
    transformer.initial_transformations(output_file_path=output_file_csv)

    print("Starting feature engineering...")
    transformer.preprocess_features()

    transformer.save_merged_data_csv("data/processed/pre_processed_data.csv", sep=";")

    transformer.save_processed_data(
        X_path='data/final/X_features.joblib',
        y_path='data/final/y_target.joblib',
        tfidf_path='data/models/tfidf_vectorizer.joblib',
        svd_tfidf_path='data/models/svd_tfidf.joblib',
        svd_embeddings_path='data/models/svd_embeddings.joblib'
    )


    # Load Final Data with Embeddings
    X = joblib.load('data/final/X_features.joblib')
    y = joblib.load('data/final/y_target.joblib')

    print(f"Shape von X: {X.shape}")
    print(f"Shape von y: {y.shape}")

    print(X[:5, :10].toarray())

if __name__ == "__main__":
    main()
