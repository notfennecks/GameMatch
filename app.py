import streamlit as st

from recommender import initialize_recommender, recommend_games

@st.cache_resource
def load_recommender():
    """
    Load and cache the GameMatch recommendation system.
    
    Streamlit will reuse these objects instead of rebuilding
    the model every time the page reruns.
    """
    return initialize_recommender()

games, tfidf, tfidf_matrix = load_recommender()

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

if "selected_games" not in st.session_state:
    st.session_state.selected_games = []

# ------------------------------------------------------------
# GAME SEARCH
# ------------------------------------------------------------

# Allow the user to select multiple games.
search_query = st.text_input(
    "Search for a game",
    placeholder = "Search for games..."
) 

selected_game = None

if search_query:
    
    matches = games[
        games["name"].str.contains(
            search_query,
            case=False,
            na=False
        )
    ]
    
    matching_names = (
        matches["name"]
        .drop_duplicates()
        .head(10)
        .tolist()
    )
    
    if matching_names:
        selected_game = st.selectbox(
            label="Select a game",
            options=matching_names,
            index=None,
            placeholder="Choose a game..."
        )
        if selected_game:
            if st.button("+ Add game"):
                if selected_game not in st.session_state.selected_games:
                    st.session_state.selected_games.append(
                        selected_game
                    )
                    
                    st.success(
                        f"Added {selected_game}"
                    )
                else:
                    st.warning(
                        f"{selected_game} is already added."
                    )
        
    else:
        
        st.info("No games found.")
        
# ------------------------------------------------------------
# SELECTED GAMES
# ------------------------------------------------------------
        
st.subheader("Your Games")

if st.session_state.selected_games:
    
    for game in st.session_state.selected_games:
        st.write(f" {game}")
        
    if st.button("Clear All"):
        st.session_state.selected_games = []
        st.rerun()
else:
    
    st.write("No games selected yet.")
    
# ------------------------------------------------------------
# GET RECOMMENDATIONS
# ------------------------------------------------------------
        
if st.button("Find Recommendations"):
    
    if not st.session_state.selected_games:
        
        st.warning(
            "Add at least one game first."
        )
        
    else:
        recommendations = recommend_games(
            st.session_state.selected_games,
            games,
            tfidf_matrix,
            num_recommendations=5
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