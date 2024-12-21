import pandas as pd


# To-Do: Implement Data Transformation in Script/Class
# while transforming: save data as raw, processed, final e.g.


class DataTransformer:
    def __init__(self, reviews_df: pd.DataFrame, metadata_df: pd.DataFrame):
        self.reviews_df = reviews_df
        self.metadata_df = metadata_df
        self.merged_df = None

    def preprocess(self):
        pass

    def handle_missing_values(self) -> None:
        self.reviews_df = self.reviews_df.dropna(subset=["text"])
        self.reviews_df["verified_purchase"] = self.reviews_df[
            "verified_purchase"
        ].fillna(False)
        self.metadata_df["main_category"] = self.metadata_df["main_category"].fillna(
            "Unknown"
        )
        self.metadata_df["store"] = self.metadata_df["store"].fillna("Unknown")
        self.metadata_df = self.metadata_df.dropna(subset=["average_rating"])

    def remove_duplicates(self) -> None:
        self.reviews_df = self.reviews_df.drop_duplicates(subset=["user_id", "asin"])

    def merge_and_rename(self) -> None:
        meta_selected_columns = [
            "parent_asin",
            "main_category",
            "title",
            "average_rating",
            "rating_number",
            "store",
            "bought_together",
        ]
        rename_columns = {
            "title_x": "review_title",
            "title_y": "product_title",
            "rating": "review_rating",
            "text": "review_text",
            "asin": "product_asin",
            "parent_asin": "product_parent_asin",
            "user_id": "review_user_id",
            "helpful_vote": "helpful_votes",
            "verified_purchase": "is_verified_purchase",
            "main_category": "product_main_category",
            "average_rating": "product_average_rating",
            "rating_number": "product_rating_count",
            "store": "product_store",
            "bought_together": "products_bought_together",
        }
        self.merged_df = self.reviews_df.merge(
            self.metadata_df[meta_selected_columns], on="parent_asin", how="left"
        ).rename(columns=rename_columns)

        return self.merged_df

    def save_merged_data_csv(self, filepath: str, sep: str = ";") -> None:
        self.merged_df.to_csv(filepath, index=False, sep=sep)
