import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import joblib
import nltk
from nltk.corpus import stopwords, opinion_lexicon
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# To-Do: Implement Data Transformation in Script/Class
# while transforming: save data as raw, processed, final e.g.


class DataTransformer:

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())

    def __init__(self, reviews_df: pd.DataFrame, metadata_df: pd.DataFrame):
        self.reviews_df = reviews_df
        self.metadata_df = metadata_df
        self.merged_df = None
        self.df_clean = None
        self.X = None
        self.y = None
        # Initialise models
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.svd_tfidf = TruncatedSVD(n_components=150, random_state=42)
        self.svd_embeddings = TruncatedSVD(n_components=150, random_state=42)
        self.embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')


    def preprocess(self):
        self.handle_missing_values()
        self.remove_duplicates()
        self.merge_and_rename()
        self.lemmatize_reviews()
        self.create_full_review()
        self.calculate_text_features()
        self.calculate_sentiment_features()
        self.log_transform_helpful_votes()
        self.handle_inf_nan_values()
        self.scale_numerical_features()
        self.vectorize_text()
        self.generate_embeddings()
        self.reduce_dimensions()
        self.combine_features()
        self.prepare_target()

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
            #"store",
            #"bought_together",
        ]
        rename_columns = {
            "title_x": "review_title",
            "title_y": "product_title",
            "rating": "review_rating",
            "text": "review_text",
            "asin": "product_asin",
            "parent_asin": "product_parent_asin",
            "user_id": "review_user_id",
            #"helpful_vote": "helpful_votes",
            "verified_purchase": "is_verified_purchase",
            "main_category": "product_main_category",
            "average_rating": "product_average_rating",
            "rating_number": "product_rating_count",
            #"store": "product_store",
            #"bought_together": "products_bought_together",
        }
        self.merged_df = self.reviews_df.merge(
            self.metadata_df[meta_selected_columns], on="parent_asin", how="left"
        ).rename(columns=rename_columns)

        # Do we need helpfule_votes? We don't have them for new reviews
        self.merged_df = self.merged_df[self.merged_df['product_main_category'] == 'Musical Instruments'][[
                        'review_rating', 'review_title', 'review_text',
                        'is_verified_purchase', 'product_title', 'product_average_rating',
                        'product_rating_count'
                    ]].dropna(subset=['review_title', 'review_text'])

        return self.merged_df

    def save_merged_data_csv(self, filepath: str, sep: str = ";") -> None:
        self.merged_df.to_csv(filepath, index=False, sep=sep)

    # Need for Feature-Engineering
    def lemmatize_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(word, 'v') for word in tokens]
        return ' '.join(lemmatized_tokens)

    def count_sentiment_words(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        pos_count = sum(1 for word in tokens if word in self.positive_words)
        neg_count = sum(1 for word in tokens if word in self.negative_words)
        return pos_count, neg_count

    def lemmatize_reviews(self):
        self.merged_df['review_title_lemmatized'] = self.merged_df['review_title'].apply(self.lemmatize_text)
        self.merged_df['review_text_lemmatized'] = self.merged_df['review_text'].apply(self.lemmatize_text)

    def create_full_review(self):
        self.merged_df['full_review'] = self.merged_df['review_title_lemmatized'] + ' ' + self.merged_df['review_text_lemmatized']

    def calculate_text_features(self):
        self.merged_df['word_count'] = self.merged_df['full_review'].apply(lambda x: len(x.split()))
        self.merged_df['char_count'] = self.merged_df['full_review'].apply(lambda x: len(x))
        self.merged_df['avg_word_length'] = self.merged_df['char_count'] / self.merged_df['word_count']

    def calculate_sentiment_features(self):
        sentiment_counts = self.merged_df['full_review'].apply(lambda x: pd.Series(self.count_sentiment_words(x)))
        sentiment_counts.columns = ['positive_word_count', 'negative_word_count']
        self.merged_df = pd.concat([self.merged_df, sentiment_counts], axis=1)

    def log_transform_helpful_votes(self):
        # Falls 'helpful_votes' nicht vorhanden ist, überspringe diesen Schritt oder setze Standardwert
        if 'helpful_votes' in self.merged_df.columns:
            self.merged_df['log_helpful_votes'] = np.log1p(self.merged_df['helpful_votes'])
        else:
            self.merged_df['log_helpful_votes'] = 0  # Oder ein anderer angemessener Wert

    def handle_inf_nan_values(self):
        numerical_features = [
            'word_count', 'char_count', 'avg_word_length',
            'positive_word_count', 'negative_word_count',
            'product_average_rating', 'product_rating_count', 'log_helpful_votes'
        ]
        # Ersetze Inf-Werte durch NaN
        self.merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Entferne Zeilen mit word_count == 0
        self.merged_df = self.merged_df[self.merged_df['word_count'] != 0]
        # Imputiere fehlende Werte mit dem Median
        self.merged_df[numerical_features] = self.merged_df[numerical_features].fillna(
            self.merged_df[numerical_features].median())

    def scale_numerical_features(self):
        numerical_features = [
            'word_count', 'char_count', 'avg_word_length',
            'positive_word_count', 'negative_word_count',
            'product_average_rating', 'product_rating_count', 'log_helpful_votes'
        ]
        self.merged_df[numerical_features] = self.scaler.fit_transform(
            self.merged_df[numerical_features]).astype(np.float32)

    def vectorize_text(self):
        self.tfidf_features = self.tfidf_vectorizer.fit_transform(
            self.merged_df['full_review']).astype(np.float32)

    def generate_embeddings(self):
        embeddings = self.embedding_model.encode(
            self.merged_df['full_review'].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype(np.float32)
        self.embeddings_sparse = sparse.csr_matrix(embeddings)

    def reduce_dimensions(self):
        # Dimensionsreduktion für Embeddings
        embeddings_reduced = self.svd_embeddings.fit_transform(self.embeddings_sparse)
        self.embeddings_reduced_sparse = sparse.csr_matrix(embeddings_reduced.astype(np.float32))
        # Dimensionsreduktion für TF-IDF-Features
        tfidf_reduced = self.svd_tfidf.fit_transform(self.tfidf_features)
        self.tfidf_reduced_sparse = sparse.csr_matrix(tfidf_reduced.astype(np.float32))

    def combine_features(self):
        numerical_features = [
            'word_count', 'char_count', 'avg_word_length',
            'positive_word_count', 'negative_word_count',
            'product_average_rating', 'product_rating_count'
        ]
        numeric_features_values = self.merged_df[numerical_features].values.astype(np.float32)
        categorical_features = self.merged_df['is_verified_purchase'].astype(int).values.reshape(-1, 1).astype(np.float32)
        # Konvertiere numerische und kategorische Features in Sparse-Matrizen
        numeric_sparse = sparse.csr_matrix(numeric_features_values)
        categorical_sparse = sparse.csr_matrix(categorical_features)
        # Kombiniere alle Features
        self.X = sparse.hstack([
            self.tfidf_reduced_sparse,
            self.embeddings_reduced_sparse,
            numeric_sparse,
            categorical_sparse
        ]).astype(np.float32)

    def prepare_target(self):
        self.y = self.merged_df['review_rating'].astype(np.int32).values

    def save_processed_data(self, X_path='X_features.joblib', y_path='y_target.joblib',
                            tfidf_path='tfidf_vectorizer.joblib',
                            svd_tfidf_path='svd_tfidf.joblib',
                            svd_embeddings_path='svd_embeddings.joblib'):
        # Speichere die Feature-Matrix und das Target
        joblib.dump(self.X, X_path, compress=9)
        joblib.dump(self.y, y_path, compress=9)
        # Speichere die Modelle
        joblib.dump(self.tfidf_vectorizer, tfidf_path, compress=9)
        joblib.dump(self.svd_tfidf, svd_tfidf_path, compress=9)
        joblib.dump(self.svd_embeddings, svd_embeddings_path, compress=9)
        print("Alle verarbeiteten Daten und Modelle wurden erfolgreich gespeichert.")

