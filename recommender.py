from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing import prepare_data

import numpy as np


# Load and preprocess the Steam dataset
games = prepare_data()

#Term Frequency-Inverse Document Frequency
#Creates the TF-IDF Vectorizer
tfidf = TfidfVectorizer()

#Convert each game's combined features into a numerical vector representation
tfidf_matrix = tfidf.fit_transform(
    games["combined_features"]
)

def recommend_games(game_name, num_recommendations=10):
    """
    Recommend games similar to the provided game.

    Args:
        game_name (str): Name of the game to search for.
        num_recommendations (int): Number of recommendations to return.
    """

    # Find the game by name, ignoring capitalization.
    matches = games[
        games["name"].str.lower() == game_name.lower()
    ]

    # Handle a game that does not exist in the dataset.
    if matches.empty:
        print(f"Game '{game_name}' not found.")
        return

    # Get the DataFrame index of the selected game.
    game_index = matches.index[0]

    # Retrieve the game's TF-IDF feature vector.
    game_vector = tfidf_matrix[game_index]

    # Compare the selected game against every game in the dataset.
    similarity_scores = cosine_similarity(
        game_vector,
        tfidf_matrix
    ).flatten()

    # Sort games from most similar to least similar.
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Remove the selected game itself and keep only the
    # requested number of recommendations.
    top_indices = [
        index for index in sorted_indices
        if index != game_index
    ][:num_recommendations]

    #Create a DataFrame containing the recommended games and their similarity scores.
    recommendations = games.iloc[top_indices][
        ["appid", "name", "genres", "tags"]
    ].copy()
    
    #Add the calculated similarity score for each recommendation.
    recommendations["similarity_score"] = [
        similarity_scores[index]
        for index in top_indices
    ]
    
    return recommendations
        
recommendations = recommend_games(
    "Stardew Valley",
    10
)

print(recommendations)