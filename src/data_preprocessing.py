# ============================================================
# GameMatch - Data Preprocessing
#
# Downloads the Steam dataset and prepares the raw data for
# use by the GameMatch recommendation engine.
# ============================================================

import ast
import shutil
from pathlib import Path

import kagglehub
import pandas as pd


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DATASET = "artermiloff/steam-games-dataset"
FILE_NAME = "games_march2025_full.csv"

DATA_DIR = Path("data/raw")
LOCAL_FILE = DATA_DIR / FILE_NAME


# ------------------------------------------------------------
# 1. DOWNLOAD DATASET
# ------------------------------------------------------------

def download_dataset():
    """
    Download the Steam dataset from Kaggle and copy it into
    GameMatch's local data/raw directory.

    Returns:
        Path: Location of the local CSV file.
    """

    # Create the local data directory if it does not exist.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Only download/copy the dataset if we do not already
    # have a local copy.
    if not LOCAL_FILE.exists():

        kaggle_path = kagglehub.dataset_download(DATASET)

        source_file = Path(kaggle_path) / FILE_NAME

        shutil.copy2(source_file, LOCAL_FILE)

    return LOCAL_FILE


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

def load_data(file_path):
    """
    Load the Steam CSV file into a Pandas DataFrame.

    Args:
        file_path (Path): Location of the CSV file.

    Returns:
        DataFrame: Raw Steam game data.
    """

    return pd.read_csv(file_path)


# ------------------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------------------

def clean_data(df):
    """
    Clean and prepare the raw Steam dataset.

    Converts structured strings back into Python objects,
    removes games without names, and handles missing
    descriptions.

    Args:
        df (DataFrame): Raw Steam dataset.

    Returns:
        DataFrame: Cleaned Steam dataset.
    """

    # Work on a copy so the original DataFrame is preserved.
    games = df.copy()

    # These columns contain Python lists stored as strings.
    list_columns = [
        "developers",
        "publishers",
        "categories",
        "genres"
    ]

    # Convert the strings back into actual Python lists.
    for column in list_columns:
        games[column] = games[column].apply(ast.literal_eval)

    # Tags are primarily dictionaries stored as strings.
    # Games without tags may contain an empty list.
    games["tags"] = games["tags"].apply(ast.literal_eval)

    # Remove games without names because they cannot be
    # searched for or displayed by GameMatch.
    games = games.dropna(subset=["name"])

    # Reset the index after removing rows.
    games = games.reset_index(drop=True)

    # Preserve games without descriptions by replacing
    # missing descriptions with an empty string.
    games["short_description"] = (
        games["short_description"].fillna("")
    )

    return games


# ------------------------------------------------------------
# 4. FEATURE NORMALIZATION
# ------------------------------------------------------------

def clean_feature(feature):
    """
    Normalize an individual genre or tag.

    Examples:
        "Open World"   -> "open_world"
        "Dark Fantasy" -> "dark_fantasy"
    """

    return (
        feature
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ------------------------------------------------------------
# 5. COMBINE FEATURES
# ------------------------------------------------------------

def combine_features(row):
    """
    Combine a game's genres and tags into one text string
    that can later be processed by TF-IDF.
    """

    genres = [
        clean_feature(genre)
        for genre in row["genres"]
    ]

    tags = [
        clean_feature(tag)
        for tag in row["tag_names"]
    ]

    return " ".join(genres + tags)


# ------------------------------------------------------------
# 6. FEATURE ENGINEERING
# ------------------------------------------------------------

def create_features(games):
    """
    Create the features needed by the recommendation model.

    Args:
        games (DataFrame): Cleaned Steam dataset.

    Returns:
        DataFrame: Dataset containing engineered ML features.
    """

    # Extract tag names from Steam's tag dictionaries.
    # Games without tags receive an empty list.
    games["tag_names"] = games["tags"].apply(
        lambda x: list(x.keys()) if isinstance(x, dict) else []
    )

    # Combine genres and tags into a single text feature.
    games["combined_features"] = games.apply(
        combine_features,
        axis=1
    )

    return games


# ------------------------------------------------------------
# 7. COMPLETE PREPROCESSING PIPELINE
# ------------------------------------------------------------

def prepare_data():
    """
    Run the complete GameMatch preprocessing pipeline.

    Returns:
        DataFrame: Cleaned and feature-engineered game data.
    """

    file_path = download_dataset()

    df = load_data(file_path)

    games = clean_data(df)

    games = create_features(games)

    return games