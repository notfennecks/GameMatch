import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing import prepare_data

from rapidfuzz import process, fuzz

from pathlib import Path
import pickle

import pandas as pd
from scipy.sparse import load_npz

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "games_processed.pkl"
)
METADATA_VECTORIZER_FILE = (
    BASE_DIR / "models" / "metadata_vectorizer.pkl"
)
METADATA_MATRIX_FILE = (
    BASE_DIR / "models" / "metadata_matrix.npz"
)
DESCRIPTION_VECTORIZER_FILE = (
    BASE_DIR / "models" / "description_vectorizer.pkl"
)
DESCRIPTION_MATRIX_FILE = (
    BASE_DIR / "models" / "description_matrix.npz"
)
DEVELOPER_VECTORIZER_FILE = (
    BASE_DIR / "models" / "developer_vectorizer.pkl"
)
DEVELOPER_MATRIX_FILE = (
    BASE_DIR / "models" / "developer_matrix.npz"
)

# Load and preprocess the Steam dataset.
games = prepare_data()

def build_tfidf_model(games):
    """
    Build the TF-IDF feature matrix used by the recommendation engine.

    Args:
        games (DataFrame): Preprocessed Steam game data.

    Returns:
        TfidfVectorizer: Fitted TF-IDF vectorizer.
        sparse matrix: Numerical representation of each game.
    """

    #Genres + tags
    metadata_tfidf = TfidfVectorizer()

    # Learn the vocabulary and convert each game's features
    # into a numerical vector.
    metadata_matrix = metadata_tfidf.fit_transform(
        games["combined_features"]
    )
    
    #Short descriptions
    description_tfidf = TfidfVectorizer(
        stop_words="english"
    )
    
    description_matrix = description_tfidf.fit_transform(
        games["short_description"]
    )
    
    #Developers
    developer_tfidf = TfidfVectorizer()
    
    developer_matrix = developer_tfidf.fit_transform(
        games["developer_features"]
    )

    return (metadata_tfidf, metadata_matrix, description_tfidf, description_matrix, developer_tfidf, developer_matrix)

def find_game(game_name, games):
    """
    Find a game in the dataset using a case-insensitive search.

    Args:
        game_name (str): Name of the game to find.

    Returns:
        int or None: DataFrame index of the game if found.
    """

    matches = games[
        games["name"].str.lower() == game_name.lower()
    ]

    if matches.empty:
        return None

    return matches.index[0]

def recommend_games(game_names, games, metadata_matrix, description_matrix, developer_matrix, num_recommendations=5):
    """
    Recommend games based on one or more games the user likes.

    Args:
        games (DataFrame): Preprocessed Steam game data.
        tfidf_matrix (sparse matrix): TF-IDF feature matrix for all games.
        num_recommendations (int): Number of recommendations to return.

    Returns:
        DataFrame: Recommended games and similarity scores.
    """

    game_indices = []

    # Find each selected game in the dataset.
    for game_name in game_names:

        game_index = find_game(game_name, games)

        if game_index is None:
            print(f"Game '{game_name}' not found.")
            continue

        game_indices.append(game_index)

    # Stop if none of the requested games were found.
    if not game_indices:
        print("None of the selected games were found.")
        return None

    # Retrieve the TF-IDF vectors for the selected games.
    
    #Genres + tags profile
    metadata_profile = np.asarray(
        metadata_matrix[game_indices].mean(axis=0)
    )
    #Description profile
    description_profile = np.asarray(
        description_matrix[game_indices].mean(axis=0)
    )
    #Developer profile
    developer_profile = np.asarray(
        developer_matrix[game_indices].mean(axis=0)
    )

    # Compare the user profile against every game.
    metadata_similarity = cosine_similarity(
        metadata_profile,
        metadata_matrix
    ).flatten()
    
    description_similarity = cosine_similarity(
        description_profile,
        description_matrix
    ).flatten()
    
    developer_similarity = cosine_similarity(
        developer_profile,
        developer_matrix
    ).flatten()
    
    similarity_scores = (
        metadata_similarity * 0.65
        + description_similarity * 0.25
        + developer_similarity * 0.10
    )

    # Sort games from most similar to least similar.
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Remove games the user already selected.
    top_indices = [
        index
        for index in sorted_indices
        if index not in game_indices
    ][:num_recommendations]

    # Create the recommendation results.
    recommendations = games.iloc[top_indices][
        [
            "appid",
            "name",
            "genres",
            "developers",
            "short_description",
            "header_image"
        ]
    ].copy()

    # Add each game's similarity score.
    recommendations["similarity_score"] = [
        similarity_scores[index]
        for index in top_indices
    ]

    return recommendations

def initialize_recommender():
    """
    Load the game data and recommendation model
    """
    games = pd.read_pickle(PROCESSED_FILE)
    
    #Load metadata vectorizer
    with open(METADATA_VECTORIZER_FILE, "rb") as file:
        metadata_tfidf = pickle.load(file)
    #Load description vectorizer    
    with open(DESCRIPTION_VECTORIZER_FILE, "rb") as file:
        description_tfidf = pickle.load(file)
    #Load developer vectorizer
    with open(DEVELOPER_VECTORIZER_FILE, "rb") as file:
            developer_tfidf = pickle.load(file)
    
    #Load sparse matrices
    metadata_matrix = load_npz(
        METADATA_MATRIX_FILE
    )
    description_matrix = load_npz(
        DESCRIPTION_MATRIX_FILE
    )
    developer_matrix = load_npz(
        DEVELOPER_MATRIX_FILE
    )       
        
    
    return (
        games,
        metadata_tfidf,
        metadata_matrix,
        description_tfidf,
        description_matrix,
        developer_tfidf,
        developer_matrix
    )

def search_games(query, game_names, limit=10):
    """
    Search for games using both substring and fuzzy matching.

    Args:
        query (str): User's search text
        games (DataFrame): Steam game dataset
        limit (int): Maximum number of results.
        
    Returns:
        list: Matching game titles.
    """
    
    #Normalize the user's input.
    query = query.strip()
    
    if not query:
        return []
    
    substring_matches = [
        name
        for name in game_names
        if query.lower() in name.lower()
    ]
    
    if len(substring_matches) >= limit:
        return substring_matches[:limit]
    
    fuzzy_matches = process.extract(
        query,
        game_names,
        scorer=fuzz.WRatio,
        limit=limit
    )
    
    results = substring_matches.copy()
    
    for name, score, _ in fuzzy_matches:
        if score >= 60 and name not in results:
            results.append(name)
            
            if len(results) >= limit:
                break
            
    return results

if __name__ == "__main__":
    
    games, tfidf, tfidf_matrix = initialize_recommender()
        
