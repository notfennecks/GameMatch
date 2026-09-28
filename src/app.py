import streamlit as st

from recommender import initialize_recommender, recommend_games, search_games

@st.cache_resource
def load_recommender():
    """
    Load and cache the GameMatch recommendation system.
    
    Streamlit will reuse these objects instead of rebuilding
    the model every time the page reruns.
    """
    return initialize_recommender()

(
    games, 
    metadata_tfidf,
    metadata_matrix, 
    description_tfidf, 
    description_matrix,
    developer_tfidf,
    developer_matrix
) = load_recommender()

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

MAX_GAMES = 5

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
    
    matching_names = search_games(search_query, games, limit=10)
    
    if matching_names:
        selected_game = st.selectbox(
            label="Select a game",
            options=matching_names,
            index=None,
            placeholder="Choose a game..."
        )
        if selected_game:
            if st.button("+ Add game"):
                
                #Prevent duplicate selections.
                if selected_game in st.session_state.selected_games:
                    
                    st.warning( f"{selected_game} is already added.")
                #Prevent user from selecting more than the maximum.
                elif len(st.session_state.selected_games) >= MAX_GAMES:
                    st.warning(f"You can select up to {MAX_GAMES} games.")
                    
                #Add the game if it passes both checks.
                else:
                    st.session_state.selected_games.append(selected_game)
                    st.rerun()
        
# ------------------------------------------------------------
# SELECTED GAMES
# ------------------------------------------------------------
        
st.subheader("Your Games")
st.caption(f"{len(st.session_state.selected_games)} / {MAX_GAMES} games selected")

if st.session_state.selected_games:
    
    for game in st.session_state.selected_games:
        #Create two columns:
        #One for the game title and one for the remove button.
        game_column, remove_column = st.columns([4, 1])
        with game_column:
            st.write(f"🎮 {game}")
        with remove_column:
            if st.button("Remove", key=f"remove_{game}"):
                st.session_state.selected_games.remove(game)
                st.rerun()
        
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
            metadata_matrix,
            description_matrix,
            developer_matrix,
            num_recommendations=6
        )
        
        st.subheader("Recommended Games")
        
        columns = st.columns(2)

        for index, (_, game) in enumerate(
            recommendations.iterrows()
        ):

            column = columns[index % 2]

            with column:

                with st.container(border=True):

                    st.image(
                        game["header_image"],
                        use_container_width=True
                    )

                    st.subheader(game["name"])

                    if isinstance(game["genres"], list):
                        genres = ", ".join(game["genres"])
                    else:
                        genres = str(game["genres"])

                    st.caption(genres)

                    match_percentage = int(
                        game["similarity_score"] * 100
                    )

                    st.write(
                        f"**{match_percentage}% Match Score**"
                    )
        
        
# streamlit run src/app.py