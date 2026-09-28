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

    print("Building TF-IDF model...")

    tfidf, tfidf_matrix = build_tfidf_model(games)

    print("Saving processed games...")

    games.to_pickle(
        PROCESSED_DIR / "games_processed.pkl"
    )

    print("Saving TF-IDF vectorizer...")

    with open(
        MODEL_DIR / "tfidf_vectorizer.pkl",
        "wb"
    ) as file:
        pickle.dump(tfidf, file)

    print("Saving TF-IDF matrix...")

    save_npz(
        MODEL_DIR / "tfidf_matrix.npz",
        tfidf_matrix
    )

    print("Artifacts successfully created.")
    
if __name__ == "__main__":
    create_artifacts()
