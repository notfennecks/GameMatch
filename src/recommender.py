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
VECTORIZER_FILE = (
    BASE_DIR
    / "models"
    / "tfidf_vectorizer.pkl"
)
MATRIX_FILE = (
    BASE_DIR
    / "models"
    / "tfidf_matrix.npz"
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

    # Create the TF-IDF vectorizer.
    tfidf = TfidfVectorizer()

    # Learn the vocabulary and convert each game's features
    # into a numerical vector.
    tfidf_matrix = tfidf.fit_transform(
        games["combined_features"]
    )

    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = build_tfidf_model(games)

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

def recommend_games(game_names, games, tfidf_matrix, num_recommendations=5):
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
    selected_vectors = tfidf_matrix[game_indices]

    # Average the vectors to create a user preference profile.
    user_profile = np.asarray(
        selected_vectors.mean(axis=0)
    )

    # Compare the user profile against every game.
    similarity_scores = cosine_similarity(
        user_profile,
        tfidf_matrix
    ).flatten()

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
    
    with open(VECTORIZER_FILE, "rb") as file:
        tfidf = pickle.load(file)
        
    tfidf_matrix = load_npz(MATRIX_FILE)
    
    return games, tfidf, tfidf_matrix

def search_games(query, games, limit=10):
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
    
    #Get unique game names from the dataset.
    game_names = games["name"].dropna().drop_duplicates().tolist()
    
    #Fist look for normal substring matches.
    substring_matches = [
        name
        for name in game_names
        if query.lower() in name.lower()
    ]
    
    #If we already have enough good subtring matches,
    #return those first.
    if len(substring_matches) >= limit:
        return substring_matches[:limit]
    
    #Use fuzzy matching to find titles similar to the query
    fuzzy_matches = process.extract(
        query,
        game_names,
        scorer=fuzz.WRatio,
        limit=limit
    )
    
    results = substring_matches.copy()
    
    for name, score, _ in fuzzy_matches:
        
        #ignore very weak fuzzy matches.
        if score >= 60 and name not in results:
            results.append(name)
        
        if len(results) >= limit:
            break
        
    return results

if __name__ == "__main__":
    
    games, tfidf, tfidf_matrix = initialize_recommender()
        
