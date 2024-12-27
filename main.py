from src.core.data.data_loader import DataLoader
import joblib

def main():
    
    data_loader = DataLoader(reviews_file='', metadata_file='')
    
    try:
        X, y, svd_embeddings = data_loader.load_processed_data(
            X_path='data/processed/X_features_reduced.joblib', 
            y_path='data/processed/y_target_reduced.joblib', 
            svd_embeddings_path='data/processed/svd_embeddings_reduced.joblib'
        )
    except FileNotFoundError as e:
        print(f"Fehler beim Laden der Datei: {e}")
        return
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        return

    # DF zum weiterarbeiten 
    print(f"Feature-Matrix X: {X.shape}")
    print(f"Zielvariable y: {y.shape}")

if __name__ == "__main__":
    main()
