from src.core.data.data_loader import DataLoader
from src.core.data.data_transformer import DataTransformer


def main():
    # Data pre-processing
    reviews_file = "./data/raw/Musical_Instruments.jsonl.gz"
    metadata_file = "./data/raw/meta_Musical_Instruments.jsonl.gz"
    output_file_csv = "./data/raw/merged_reviews_metadata_renamed.csv"

    loader = DataLoader(reviews_file, metadata_file)
    print("loader starts")
    loader.load_data(sample_size=150000)
    loader.drop_columns()

    transformer = DataTransformer(loader.reviews_df, loader.metadata_df)
    print("transformer starts")
    transformer.handle_missing_values()
    transformer.remove_duplicates()
    merged_df = transformer.merge_and_rename()

    # Save merged_df as csv
    transformer.save_merged_data_csv(output_file_csv, sep=";")

    print(merged_df.head())
    print(merged_df.shape)


if __name__ == "__main__":
    main()
