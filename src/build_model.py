from pathlib import Path
import pickle

from scipy.sparse import save_npz

from data_preprocessing import prepare_data
from recommender import build_tfidf_model

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def create_artifacts():

    print("Preparing dataset...")
    games = prepare_data()
    
    #-------------------------------------
    print("Building TF-IDF model...")
    (metadata_tfidf, metadata_matrix, description_tfidf, description_matrix, developer_tfidf, developer_matrix) = build_tfidf_model(games)

    #-------------------------------------
    print("Saving processed games...")
    
    games_for_app = games[
        [
            "appid",
            "name",
            "genres",
            "developers",
            "short_description",
            "header_image"
        ]
    ].copy()
    

    games_for_app.to_pickle(
        PROCESSED_DIR / "games_processed.pkl"
    )
    
    #-------------------------------------
    print("Saving TF-IDF vectorizer...")

    with open(
        MODEL_DIR / "metadata_vectorizer.pkl",
        "wb"
    ) as file:
        pickle.dump(metadata_tfidf, file)
        
    with open(
        MODEL_DIR / "description_vectorizer.pkl",
        "wb"
    ) as file:
        pickle.dump(description_tfidf, file)
        
    with open(
        MODEL_DIR / "developer_vectorizer.pkl",
        "wb"
    ) as file:
        pickle.dump(developer_tfidf, file)

    print("Saving TF-IDF matrix...")

    save_npz(
        MODEL_DIR / "metadata_matrix.npz",
       metadata_matrix
    )
    
    save_npz(
            MODEL_DIR / "description_matrix.npz",
           description_matrix
        )
    
    save_npz(
            MODEL_DIR / "developer_matrix.npz",
           developer_matrix
        )

    print("Artifacts successfully created.")
    
if __name__ == "__main__":
    create_artifacts()
