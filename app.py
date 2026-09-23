import streamlit as st

from recommender import games, recommend_games


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="GameMatch",
    page_icon="🎮",
    layout="wide"
)


# ------------------------------------------------------------
# PAGE HEADER
# ------------------------------------------------------------

st.title("🎮 GameMatch")

st.write(
    "Select games you enjoy and GameMatch will recommend "
    "similar games based on their genres and Steam tags."
)


# ------------------------------------------------------------
# GAME SELECTION
# ------------------------------------------------------------

# Get all game names from the dataset and sort them alphabetically.
game_names = sorted(
    games["name"].dropna().unique()
)

# Allow the user to select multiple games.
selected_games = st.multiselect(
    "What games do you like?",
    options=game_names,
    placeholder="Search for games..."
)


# ------------------------------------------------------------
# RECOMMENDATIONS
# ------------------------------------------------------------

if st.button("Find Recommendations"):

    if not selected_games:

        st.warning(
            "Select at least one game first."
        )

    else:

        recommendations = recommend_games(
            selected_games,
            num_recommendations=10
        )

        st.subheader("Recommended Games")

        st.dataframe(
            recommendations[
                [
                    "name",
                    "genres",
                    "similarity_score"
                ]
            ],
            hide_index=True
        )
        
# streamlit run app.py