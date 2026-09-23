import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing import prepare_data

# Load and preprocess the Steam dataset.
games = prepare_data()

def build_model(games):
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

tfidf, tfidf_matrix = build_model(games)

def find_game(game_name):
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

def recommend_games(game_names, num_recommendations=10):
    """
    Recommend games based on one or more games the user likes.

    Args:
        game_names (list): Names of games the user likes.
        num_recommendations (int): Number of recommendations to return.

    Returns:
        DataFrame: Recommended games and similarity scores.
    """

    game_indices = []

    # Find each selected game in the dataset.
    for game_name in game_names:

        game_index = find_game(game_name)

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
            "genres"
        ]
    ].copy()

    # Add each game's similarity score.
    recommendations["similarity_score"] = [
        similarity_scores[index]
        for index in top_indices
    ]

    return recommendations

if __name__ == "__main__":

    favorite_games = [
        "ELDEN RING",
        "The Witcher 3: Wild Hunt",
        "The Elder Scrolls V: Skyrim"
    ]

    recommendations = recommend_games(
        favorite_games,
        num_recommendations=10
    )

    if recommendations is not None:
        print(
            recommendations[
                ["name", "similarity_score"]
            ].to_string(index=False)
        )
 
 
       
recommendations = recommend_games(
    ["Marvel Rivals"],
    num_recommendations=3
)
print(
    recommendations[
        ["name", "similarity_score"]
    ].to_string(index=False)
)