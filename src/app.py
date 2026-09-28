import streamlit as st

from recommender import initialize_recommender, recommend_games, search_games
from streamlit_searchbox import st_searchbox

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

game_names = (
    games["name"].dropna().drop_duplicates().tolist()
)

def search_game_names(searchterm):
    if not searchterm:
        return []
    
    return search_games(
        searchterm,
        game_names,
        limit=10
    )

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
st.markdown("""
### Find your next favorite game
Tell GameMatch what you love playing and we'll find similar games.
<style>
    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)

MAX_GAMES = 5

if "selected_games" not in st.session_state:
    st.session_state.selected_games = []

# ------------------------------------------------------------
# GAME SEARCH
# ------------------------------------------------------------

# Allow the user to select multiple games.
# Allow the user to search for and select a game.

selected_game = st_searchbox(
    search_game_names,
    placeholder="Search for a game...",
    label="Search for a game",
    key="game_search"
)

if selected_game:

    # Prevent duplicate selections.
    if selected_game in st.session_state.selected_games:
        pass
        #st.warning(f"{selected_game} is already added.")

    # Prevent user from selecting more than the maximum.
    elif len(st.session_state.selected_games) >= MAX_GAMES:
        st.warning(f"You can select up to {MAX_GAMES} games.")

    # Add the game if it passes both checks.
    else:
        st.session_state.selected_games.append(selected_game)
        st.rerun()
        
# ------------------------------------------------------------
# SELECTED GAMES
# ------------------------------------------------------------
        
#st.subheader("Your Games")
#st.caption(f"{len(st.session_state.selected_games)} / {MAX_GAMES} games selected")

with st.container(border=True):
    
    st.subheader("Your Games")
    st.caption(f"{len(st.session_state.selected_games)} / "
        f"{MAX_GAMES} games selected"
    )
    if st.session_state.selected_games:
        
        for game in st.session_state.selected_games:
            game_data = games[
                games["name"].str.lower() == game.lower()
            ].iloc[0]
            #Create two columns:
            #One for the game title and one for the remove button.
            image_column, game_column, remove_column = st.columns([1, 4, 1])
            with image_column:
                if game_data["header_image"]:
                    st.image(
                        game_data["header_image"],
                        use_container_width=True
                    )
            with game_column:
                st.markdown(
                    f"""
                    <div style="
                        font-weight: 700;
                        font-size: 1.15rem;
                    ">
                        {game}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with remove_column:
                if st.button("✕", key=f"remove_{game}"):
                    st.session_state.selected_games.remove(game)
                    st.rerun()
            
        if st.button("Clear All"):
            st.session_state.selected_games = []
            st.rerun()
    else:
        
        st.info("Search above and add up to 5 games to build your taste profile.")
    
# ------------------------------------------------------------
# GET RECOMMENDATIONS
# ------------------------------------------------------------

find_recommendations = st.button(
    "✨ Find Recommendations",
    type="primary",
    use_container_width=True,
    disabled=len(st.session_state.selected_games) == 0
)
if find_recommendations:
    with st.spinner("Finding games you'll love..."):
        recommendations = recommend_games(
            st.session_state.selected_games,
            games,
            metadata_matrix,
            description_matrix,
            developer_matrix,
            num_recommendations=6
        )
    
    st.divider()  
    st.subheader("🎯 Recommended For You")
        
    columns = st.columns(2)

    for index, (_, game) in enumerate(
        recommendations.iterrows()
    ):

        column = columns[index % 2]

        with column:
            with st.container(height=500, border=True):

                steam_url = (f"https://store.steampowered.com/app/{game['appid']}/")
                
                if game["header_image"]:
                    st.markdown(
                        f"""
                        <a href="{steam_url}" target="_blank">
                            <img
                                src="{game['header_image']}"
                                style="
                                    width: 100%;
                                    border-radius: 8px;
                                    cursor: pointer;
                                "
                            >
                        </a>
                        """,
                        unsafe_allow_html=True
                    )

                st.subheader(game["name"])
                
                #Developers
                if isinstance(game["developers"], list):
                    developers = ", ".join(game["developers"])
                else:
                    developers = str(game["developers"])

                st.caption(f"Developed by {developers}")

                #Genres
                if isinstance(game["genres"], list):
                    genres = ", ".join(game["genres"])
                else:
                    genres = str(game["genres"])

                st.caption(genres)
                
                st.markdown(game["short_description"], unsafe_allow_html=True)

                #Removed because some games are very closely realted but match scores are lower
                #match_percentage = int(
                    #game["similarity_score"] * 100
                #)

                #st.write(f"**⭐ {match_percentage}% Match Score**")
                #st.progress(float(game["similarity_score"]))
                #"""
                
        
        
# streamlit run src/app.py