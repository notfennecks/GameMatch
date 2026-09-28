## **App URL: https://game-match.streamlit.app/**

## Overview
**GameMatch**
An AI powered engine that gives game recommendations (Only for games published on Steam). It is a content-based video game recommendation application that helps users discover new games based on titles they already enjoy. Users can search for and choose up to 5 titles they love and GameMatch analyzes their combined characterstics and generates a personalized list from a dataset of approx 95,000 Steam games.

The engine represents games using **3 indepenend feature groups**:

-**Genres and community tags**
-**Game description**
-**Developers**

Each feature group is transformed in a numerical representation using **TF-IDF (Term Frequency-Inverse Document Frequency)**. GameMatch then creates a preference profile from the games selected by the user and compares that profile against the Steam game catalog using **cosine similarity**.

The 3 similarity measurements are combined using a weighted scoring system:

-65% - Genres and Tags
-25% - Game Description
-10% - Developer

Games are ranked according to their combined similarity score, while games already selected by the user and duplicate titles are excluded from the final results.

GameMath was developed as an end-to-end machine learning and software engineering project using Python, pandas, scikit-learn, SciPy, RapidFuzz, and Streamlit. The project includes data preprocessing, feature engineering, model artifact generation, recommendation logic, fuzzy game search, an interaction web interface, and a deployment-ready architecture.

## Features
- **Personalized Game Recommendations** — Generates recommendations based on the combined preferences of up to five games selected by the user.
- **Large Steam Game Catalog** — Searches and recommends from a dataset containing approximately 95,000 Steam games.
- **Multi-Feature Recommendation Engine** — Compares games using genres, community tags, game descriptions, and developers rather than relying on a single feature.
- **Weighted Similarity Model** — Combines multiple cosine similarity measurements using separate weights for metadata, descriptions, and developers.
- **Multi-Game Preference Profiles** — Averages the feature representations of multiple selected games to create a single user preference profile.
- **Fuzzy Game Search** — Uses RapidFuzz to handle partial titles, capitalization differences, and minor spelling mistakes when searching for games.
- **Autocomplete Search** — Provides responsive search suggestions without loading the entire catalog of approximately 95,000 titles into the interface at once.
- **Duplicate Filtering** — Prevents selected games from appearing in recommendations and removes duplicate game titles from recommendation results.
- **Steam Integration** — Displays Steam header artwork and provides direct links to each recommended game's Steam store page using its Steam App ID.
- **Interactive Streamlit Interface** — Provides a simple web interface for searching, selecting, removing, and receiving game recommendations.
- **Precomputed Model Artifacts** — TF-IDF matrices and processed game data are generated ahead of time so the recommendation model does not need to be rebuilt every time the application starts.
- **Sparse Matrix Representation** — Uses SciPy sparse matrices to efficiently store TF-IDF representations across the large game catalog.
- **On-Demand Similarity Calculation** — Calculates similarity between the user's preference profile and the game catalog only when recommendations are requested rather than storing a massive all-pairs similarity matrix.

## System Architecture

GameMatch separates **offline data/model preparation** from the **runtime recommendation application**. The Steam dataset is processed and vectorized ahead of time, allowing the deployed application to load precomputed artifacts rather than rebuilding the recommendation model whenever the application starts.

### Architecture Overview
                   Steam Games Dataset
                    (~95,000 games)
                           │
                           ▼
                  ┌─────────────────┐
                  │ Data Processing │
                  │                 │
                  │ • Clean data    │
                  │ • Parse fields  │
                  │ • Handle nulls  │
                  │ • Build features│
                  └────────┬────────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
      ┌────────────┐ ┌────────────┐ ┌────────────┐
      │ Genres &   │ │   Short    │ │ Developer  │
      │    Tags    │ │Description │ │            │
      └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
            │              │              │
            ▼              ▼              ▼
      ┌──────────┐   ┌──────────┐   ┌──────────┐
      │  TF-IDF  │   │  TF-IDF  │   │  TF-IDF  │
      │  Matrix  │   │  Matrix  │   │  Matrix  │
      └────┬─────┘   └────┬─────┘   └────┬─────┘
           │              │              │
           └──────────────┼──────────────┘
                          │
                          ▼
               Precomputed Model Artifacts
                          │
                          ▼
               ┌─────────────────────┐
               │    Streamlit App    │
               │                     │
               │ User selects 1–5   │
               │ games they enjoy   │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Preference Profiles │
               │                     │
               │ Average selected    │
               │ game vectors        │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Cosine Similarity   │
               │                     │
               │ Profile vs. entire  │
               │ Steam catalog       │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Weighted Similarity │
               │                     │
               │ Metadata      65%   │
               │ Description   25%   │
               │ Developer     10%   │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Rank & Filter Games │
               │                     │
               │ • Sort by score     │
               │ • Remove selected   │
               │ • Remove duplicates │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │   Recommendations   │
               │                     │
               │ Top 10 games with   │
               │ artwork & details   │
               └─────────────────────┘

### Offline Processing Pipeline

The computationally expensive preprocessing and vectorization steps are performed separately from the web application.

The raw Steam dataset is cleaned and transformed into features representing each game's:

- Genres
- Community tags
- Short description
- Developer

Three separate TF-IDF models are then created for **metadata (genres and tags)**, **descriptions**, and **developers**. Their resulting sparse matrices are saved as precomputed model artifacts.

The processed game information required by the application is also stored separately. This allows GameMatch to start by loading existing artifacts instead of processing the full raw dataset and rebuilding TF-IDF matrices every time the application launches.

### Runtime Recommendation Pipeline

When a user selects games in the Streamlit application, GameMatch finds the corresponding rows within the processed dataset.

For each of the three feature spaces, the vectors belonging to the selected games are averaged to create a **user preference profile**.

GameMatch then calculates cosine similarity between each preference profile and the corresponding feature vectors for the entire game catalog.

The three similarity scores are combined using:

```text
Final Similarity =
    (Metadata Similarity × 0.65)
  + (Description Similarity × 0.25)
  + (Developer Similarity × 0.10)
```

Games are sorted by this combined similarity score. Games already selected by the user and duplicate titles are excluded, and the highest-ranked remaining games are returned as recommendations.

### Why Separate Feature Spaces?

Genres, tags, natural-language descriptions, and developer names represent different types of information.

Instead of combining every feature into a single text field, GameMatch maintains separate TF-IDF representations for:

```text
Genres + Tags  →  Metadata TF-IDF
Descriptions   →  Description TF-IDF
Developers     →  Developer TF-IDF
```

This makes it possible to independently control how strongly each type of information influences the final recommendation.

For example, genres and tags receive the largest weight because they provide strong information about gameplay style and category, while developer similarity provides an additional but smaller signal.

## Recommendation Engine

GameMatch uses a **content-based recommendation system**. Instead of relying on ratings or behavior from other users, recommendations are generated by comparing the characteristics of games the user already enjoys with other games in the Steam catalog.

The recommendation process consists of four main stages: **feature representation, preference profile generation, similarity calculation, and ranking**.

### TF-IDF Feature Representation

GameMatch uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert textual game features into numerical vectors that can be mathematically compared.

Three independent TF-IDF representations are created:

| Feature Space | Information Used | Weight |
|---|---|---:|
| Metadata | Genres + Steam community tags | 65% |
| Description | Short game descriptions | 25% |
| Developer | Game developers | 10% |

For metadata, multi-word features are normalized so they remain meaningful individual concepts.

For example:

```text
Open World   → open_world
Dark Fantasy → dark_fantasy
Souls-like   → souls_like
Online Co-Op → online_co_op
```

This prevents a feature such as `Dark Fantasy` from being treated simply as two unrelated words.

Descriptions are vectorized separately and use English stop-word filtering to reduce the influence of common words that provide little information about a game's content.

### Multi-Game Preference Profiles

GameMatch allows the user to select up to five games rather than generating recommendations from only one title.

For each feature space, the TF-IDF vectors of the selected games are averaged:

```text
Selected Games

Game A ─┐
Game B ─┼──► Average TF-IDF Vector ──► User Preference Profile
Game C ─┘
```

This process is performed independently for:

```text
Metadata Profile
Description Profile
Developer Profile
```

As a result, recommendations can reflect characteristics shared across several games the user enjoys rather than being based entirely on one game.

### Cosine Similarity

Once the preference profiles are created, GameMatch uses **cosine similarity** to compare each profile against every game in the catalog.

Cosine similarity measures the angle between two vectors:

```text
                    A · B
cosine(A, B) = ───────────────
                ||A|| × ||B||
```

Vectors pointing in similar directions represent games with similar feature patterns.

GameMatch calculates three separate similarity arrays:

```text
Metadata Profile
       │
       └──► Metadata Similarity ──┐

Description Profile              │
       │                          │
       └──► Description Similarity├──► Weighted Score

Developer Profile                │
       │                          │
       └──► Developer Similarity ─┘
```

### Weighted Similarity

The three similarity measurements are combined into a final ranking score:

```text
Final Score =
    (Metadata Similarity × 0.65)
  + (Description Similarity × 0.25)
  + (Developer Similarity × 0.10)
```

Metadata receives the largest weight because genres and community tags provide a strong representation of a game's gameplay characteristics and overall style.

Descriptions provide additional semantic information that can distinguish games that otherwise share similar genres and tags.

Developer similarity receives a smaller weight, allowing games from the same developer to receive an additional similarity signal without allowing the developer alone to dominate the recommendation.

These weights are configurable and were selected as practical starting values for the current version of GameMatch.

### Ranking and Filtering

After calculating the weighted scores, GameMatch sorts the entire game catalog from highest to lowest similarity.

The ranking process then filters the results to ensure that:

- Games selected by the user are not recommended back to them.
- Duplicate game titles are excluded.
- Only the highest-ranked unique games are returned.

The system continues through the ranked results until the requested number of unique recommendations has been found.

```text
Weighted Similarity Scores
           │
           ▼
   Sort Highest → Lowest
           │
           ▼
 Remove User-Selected Games
           │
           ▼
   Remove Duplicate Titles
           │
           ▼
  Select Top Recommendations
```

### On-Demand Similarity Calculation

GameMatch does **not** precompute similarity between every possible pair of games.

With approximately 95,000 games, an all-pairs similarity matrix would require roughly:

```text
95,000 × 95,000
≈ 9 billion similarity values
```

Instead, GameMatch creates the user's preference vectors when recommendations are requested and compares those vectors against the catalog.

Conceptually:

```text
1 user profile × ~95,000 games
```

rather than:

```text
~95,000 games × ~95,000 games
```

This significantly reduces unnecessary computation and storage while still allowing recommendations to be generated dynamically from different combinations of user-selected games.

### Similarity Scores vs. Probabilities

The internal similarity score is used to **rank recommendations**, but it is not presented as a probability or confidence percentage.

For example, a cosine-based score of `0.72` does not mean there is a "72% chance" that the user will enjoy the game. It only represents the relative similarity of the game's feature vector to the user's generated preference profile.

For this reason, GameMatch uses similarity scores internally for ranking while presenting the resulting ordered recommendations to the user.

## 🛠️ Technology Stack

GameMatch was built primarily in Python using a combination of machine learning, data processing, search, scientific computing, and web application libraries.

| Technology | Purpose |
|---|---|
| **Python** | Core programming language used throughout the application |
| **pandas** | Dataset loading, cleaning, transformation, and game metadata management |
| **scikit-learn** | TF-IDF vectorization and cosine similarity calculations |
| **SciPy** | Storage and loading of sparse TF-IDF matrices |
| **NumPy** | Numerical operations, preference profile construction, and recommendation ranking |
| **RapidFuzz** | Fuzzy game-title matching and typo-tolerant search |
| **Streamlit** | Interactive web application and user interface |
| **streamlit-searchbox** | Dynamic autocomplete game search |
| **KaggleHub** | Retrieval of the Steam dataset during the offline data preparation process |
| **Git & GitHub** | Version control, source-code management, and deployment integration |

### Python

Python serves as the primary language for the entire GameMatch pipeline, including:

- Data preprocessing
- Feature engineering
- Model generation
- Recommendation calculations
- Search functionality
- Artifact management
- Web application logic

Keeping the data pipeline, recommendation engine, and application within the Python ecosystem also allows the machine learning components to integrate directly with the Streamlit interface.

### pandas

**pandas** is used to load and process the Steam game dataset.

Its responsibilities include:

- Loading the raw game dataset
- Handling missing values
- Parsing structured columns
- Cleaning game metadata
- Creating model features
- Managing the processed game catalog
- Retrieving information for selected and recommended games

The processed dataset contains only the information required by the deployed application, reducing unnecessary runtime storage.

### scikit-learn

**scikit-learn** provides the core machine learning functionality used by the recommendation engine.

GameMatch uses:

```python
TfidfVectorizer
```

to convert game metadata, descriptions, and developer information into numerical feature vectors.

It also uses:

```python
cosine_similarity
```

to compare the generated user preference profiles against games in the Steam catalog.

### SciPy

TF-IDF produces highly sparse feature matrices because most games contain only a small subset of all possible terms.

GameMatch uses **SciPy sparse matrices** to efficiently represent this data rather than storing every matrix element as a dense value.

The generated matrices are saved as `.npz` artifacts:

```text
metadata_matrix.npz
description_matrix.npz
developer_matrix.npz
```

These matrices can then be loaded directly when the application starts.

### NumPy

**NumPy** supports numerical operations within the recommendation pipeline.

It is used when:

- Creating averaged preference vectors
- Working with similarity arrays
- Sorting similarity scores
- Ranking recommendation candidates

For example, the indices of games can be ordered by their calculated similarity scores before filtering and selecting the final recommendations.

### RapidFuzz

With approximately 95,000 game titles available, GameMatch uses **RapidFuzz** to provide typo-tolerant title matching.

This allows searches to remain useful when users enter:

- Partial titles
- Different capitalization
- Minor spelling mistakes

The search system first attempts efficient substring matching and then supplements those results with fuzzy matching when necessary.

### Streamlit

**Streamlit** provides the web interface for GameMatch.

The interface allows users to:

- Search for games
- Select up to five favorite games
- Remove selected games
- Generate recommendations
- View game artwork
- Read game descriptions
- View genres and developers
- Open recommended games directly on Steam

Streamlit session state is used to maintain the user's selected games while interacting with the application.

### streamlit-searchbox

**streamlit-searchbox** provides the autocomplete search component used by GameMatch.

Rather than sending the entire catalog of approximately 95,000 titles to a standard dropdown, search results are generated dynamically as the user types.

Only a small number of relevant results are displayed, producing a more responsive search experience.

### KaggleHub

**KaggleHub** is used by the offline data preparation pipeline to retrieve the Steam Games Dataset.

The raw dataset is used to generate the processed dataset and TF-IDF model artifacts.

The deployed recommendation application does not need to repeatedly preprocess the original dataset because these artifacts are generated ahead of time.

### Git & GitHub

**Git** is used for version control throughout development, while **GitHub** hosts the GameMatch source code and supports the deployment workflow.

Large raw source data is excluded from version control, while the repository contains the code and deployment resources required to reproduce and run the project.

## 📊 Dataset & Data Preprocessing

GameMatch uses the **Steam Games Dataset 2025** available through Kaggle. The dataset contains information for approximately **95,000 Steam games** and provides the metadata used to build the recommendation system.

The original dataset contains approximately **47 columns**, including information such as:

- Steam App ID
- Game name
- Release date
- Price
- Developers
- Publishers
- Genres
- Categories
- Community tags
- Short descriptions
- Header images
- Ratings and other Steam metadata

Not every field is required by the recommendation engine. GameMatch therefore uses a preprocessing pipeline to clean the raw data, extract relevant features, and generate a smaller runtime dataset.

### Data Pipeline

The preprocessing workflow follows the general structure:

```text
Raw Steam Dataset
        │
        ▼
   Load Dataset
        │
        ▼
 Handle Missing Data
        │
        ▼
Parse Structured Columns
        │
        ▼
 Normalize Features
        │
        ▼
Create Model Features
        │
        ▼
Build TF-IDF Matrices
        │
        ▼
Save Processed Dataset
   + Model Artifacts
```

### Parsing Structured Data

Several columns in the original CSV contain structured information stored as text, including:

```text
developers
publishers
categories
genres
tags
```

These values are parsed back into Python data structures during preprocessing.

Steam tags require additional handling because the dataset contains different representations depending on whether tag information is available.

When tags are represented as a dictionary, GameMatch extracts the tag names for use by the recommendation engine.

Conceptually:

```text
{
    "Open World": ...,
    "Action RPG": ...,
    "Dark Fantasy": ...
}

        ↓

[
    "Open World",
    "Action RPG",
    "Dark Fantasy"
]
```

Games without tag information are retained rather than removed from the dataset.

### Handling Missing Values

The preprocessing pipeline handles missing values differently depending on how important the field is to the application.

Games without a valid name cannot be searched for or displayed properly, so those records are removed.

Missing descriptions, however, do not make a game unusable. These values are replaced with empty strings so the game can still participate in the metadata and developer portions of the recommendation model.

This allows GameMatch to preserve as much of the Steam catalog as possible.

### Feature Normalization

Genres, tags, and developer information are normalized before TF-IDF vectorization.

Text is converted to lowercase and spaces or hyphens within individual features are replaced with underscores.

For example:

```text
Open World     → open_world
Action RPG     → action_rpg
Dark Fantasy   → dark_fantasy
Souls-like     → souls_like
Online Co-Op   → online_co_op
```

This ensures that multi-word concepts remain identifiable features rather than being broken into unrelated individual words.

### Metadata Feature Construction

Genres and Steam community tags are combined to create the primary metadata representation for each game.

For example:

```text
Genres:
Action
RPG

Tags:
Souls-like
Dark Fantasy
Open World

        ↓

Combined Metadata:

action rpg souls_like dark_fantasy open_world
```

This combined representation becomes the input to the metadata TF-IDF vectorizer.

Descriptions and developers remain separate because they are represented by their own TF-IDF models.

The resulting feature spaces are therefore:

```text
combined_features
        │
        └── Genres + Tags

short_description
        │
        └── Natural-language game description

developer_features
        │
        └── Normalized developer information
```

### Runtime Dataset Optimization

The original processed DataFrame initially contained nearly all of the source dataset's columns and required approximately:

```text
487 MB
```

However, most of those fields are unnecessary after the TF-IDF matrices have been generated.

GameMatch therefore stores only the columns required by the running application:

```text
appid
name
genres
developers
short_description
header_image
```

After removing unnecessary runtime columns, the processed dataset was reduced to approximately:

```text
35 MB
```

This represents a reduction of more than **90%** while preserving the information required by the recommendation interface.

The runtime dataset supports:

- Game-title search
- Recommendation results
- Genre information
- Developer information
- Game descriptions
- Steam header artwork
- Steam store links through App IDs

### Precomputed Model Artifacts

The preprocessing pipeline also generates the TF-IDF matrices and vectorizers ahead of deployment.

The generated model artifacts include:

```text
models/
├── metadata_matrix.npz
├── metadata_vectorizer.pkl
├── description_matrix.npz
├── description_vectorizer.pkl
├── developer_matrix.npz
└── developer_vectorizer.pkl
```

The optimized game catalog is stored separately:

```text
data/
└── processed/
    └── games_processed.pkl
```

Separating model generation from application runtime allows the deployed application to load the already-prepared dataset and sparse matrices instead of rebuilding them whenever the application starts.

### Maintaining Matrix Alignment

The processed game dataset and TF-IDF matrices preserve the same row ordering.

This is important because each matrix row corresponds directly to the same game in the processed DataFrame:

```text
games.iloc[0]  ↔  metadata_matrix[0]
               ↔  description_matrix[0]
               ↔  developer_matrix[0]

games.iloc[1]  ↔  metadata_matrix[1]
               ↔  description_matrix[1]
               ↔  developer_matrix[1]

                    ...
```

This alignment allows GameMatch to efficiently retrieve the appropriate vectors for selected games and map ranked similarity results back to their corresponding game information.

## ⚡ Performance & Optimization

GameMatch operates on a catalog of approximately **95,000 games**, making performance and memory usage important considerations throughout development.

Several optimizations were implemented to reduce startup time, memory usage, search latency, and unnecessary computation.

### Precomputed Model Artifacts

An early version of GameMatch performed data preprocessing and TF-IDF vectorization when the application started.

Although this worked during development, rebuilding multiple TF-IDF representations from the full dataset every time the application launched was unnecessary and increased startup time.

GameMatch now separates model preparation from application runtime.

The offline build pipeline performs:

```text
Raw Dataset
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Save Processed Data + Sparse Matrices
```

The application can then start using:

```text
Precomputed Artifacts
        │
        ▼
Load into Memory
        │
        ▼
GameMatch Ready
```

This means expensive preprocessing and vectorization only need to be performed when the dataset or model configuration changes.

### Sparse TF-IDF Matrices

TF-IDF matrices contain a large number of zero values because an individual game only contains a small subset of all terms in the vocabulary.

Storing these matrices as dense arrays would therefore waste memory.

GameMatch uses **SciPy sparse matrices** and stores them as compressed `.npz` files:

```text
metadata_matrix.npz
description_matrix.npz
developer_matrix.npz
```

This allows the recommendation engine to efficiently work with high-dimensional feature representations without storing unnecessary zero values.

### Avoiding an All-Pairs Similarity Matrix

A catalog containing approximately 95,000 games would require roughly:

```text
95,000 × 95,000
≈ 9,025,000,000
```

similarity values to precompute the similarity between every possible pair of games.

GameMatch avoids generating this matrix.

Instead, when the user requests recommendations, the selected games are used to construct three preference vectors:

```text
Metadata Profile
Description Profile
Developer Profile
```

Each profile is then compared against the corresponding catalog matrix.

Conceptually, GameMatch performs:

```text
1 preference profile × ~95,000 games
```

for each feature space instead of:

```text
~95,000 games × ~95,000 games
```

This allows recommendations to remain dynamic without requiring a massive precomputed similarity matrix.

### Optimized Runtime Dataset

The original processed dataset occupied approximately:

```text
487 MB
```

because it retained many columns required during preprocessing but unnecessary during application runtime.

The deployment dataset was reduced to only:

```text
appid
name
genres
developers
short_description
header_image
```

This reduced the processed artifact to approximately:

```text
35 MB
```

while preserving all information required by the user interface and recommendation engine.

This represents a reduction of more than **90%** in the size of the processed runtime dataset.

### Efficient Game Search

A standard dropdown containing approximately 95,000 game titles created unnecessary interface overhead.

GameMatch instead uses a dynamic autocomplete search system.

As the user types:

```text
User Input
    │
    ▼
Substring Search
    │
    ▼
Fuzzy Matching
    │
    ▼
Top Matches
    │
    ▼
Autocomplete Results
```

Only a small number of relevant game titles are returned to the interface.

The search system uses exact substring matching first and supplements those results with **RapidFuzz** fuzzy matching, allowing GameMatch to support minor spelling mistakes without rendering the entire game catalog as a dropdown.

The list of searchable game names is also prepared once after loading the processed dataset rather than being reconstructed for every search operation.

### Streamlit Resource Caching

GameMatch uses Streamlit resource caching for expensive application resources.

The processed dataset and model artifacts are loaded through:

```python
@st.cache_resource
```

This prevents the application from repeatedly loading the same matrices and dataset during normal Streamlit reruns.

Without caching, interactions such as selecting or removing a game could cause expensive resources to be unnecessarily reloaded.

### Offline Build vs. Runtime Responsibilities

GameMatch intentionally separates responsibilities between two stages:

```text
OFFLINE BUILD
─────────────
Download dataset
Clean data
Engineer features
Train TF-IDF vectorizers
Generate sparse matrices
Create optimized dataset
Save artifacts


APPLICATION RUNTIME
───────────────────
Load artifacts
Search games
Maintain user selections
Create preference profiles
Calculate similarity
Rank recommendations
Display results
```

This separation keeps the deployed application focused on recommendation generation rather than model preparation.

### Optimization Summary

The combination of these techniques allows GameMatch to work efficiently with a relatively large game catalog:

| Optimization | Purpose |
|---|---|
| Precomputed artifacts | Avoid rebuilding models at startup |
| Sparse matrices | Reduce memory and storage requirements |
| On-demand similarity | Avoid a ~9-billion-value similarity matrix |
| Reduced runtime dataset | Decrease processed data from ~487 MB to ~35 MB |
| Dynamic autocomplete | Avoid rendering ~95,000 titles at once |
| Precomputed game-name list | Reduce repeated search preparation |
| Streamlit caching | Prevent unnecessary artifact reloads |
| Offline/runtime separation | Keep deployment focused on inference |

These optimizations allow GameMatch to provide interactive recommendations while keeping the architecture relatively lightweight and reproducible.

## 📁 Project Structure

GameMatch separates data preparation, model generation, recommendation logic, and the user interface into individual modules.

```text
GameMatch/
│
├── data/
│   ├── raw/
│   │   └── games_march2025_full.csv
│   │
│   └── processed/
│       └── games_processed.pkl
│
├── models/
│   ├── metadata_matrix.npz
│   ├── metadata_vectorizer.pkl
│   ├── description_matrix.npz
│   ├── description_vectorizer.pkl
│   ├── developer_matrix.npz
│   └── developer_vectorizer.pkl
│
├── src/
│   ├── app.py
│   ├── build_model.py
│   ├── data_preprocessing.py
│   └── recommender.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### `src/app.py`

The main entry point for the GameMatch web application.

This module is responsible for the **Streamlit user interface** and connects the frontend to the recommendation engine.

Primary responsibilities include:

- Loading and caching the recommendation system
- Managing Streamlit session state
- Providing autocomplete game search
- Adding and removing selected games
- Enforcing the five-game selection limit
- Triggering recommendation generation
- Displaying recommended games
- Displaying game artwork, genres, developers, and descriptions
- Creating links to Steam store pages

The application can be launched with:

```bash
streamlit run src/app.py
```

### `src/recommender.py`

Contains the core recommendation logic used by GameMatch.

This module is responsible for:

- Loading processed game data
- Loading precomputed sparse matrices
- Loading TF-IDF vectorizers
- Finding games within the processed dataset
- Creating multi-game preference profiles
- Calculating cosine similarity
- Combining similarity signals using the weighted scoring model
- Ranking recommendation candidates
- Excluding user-selected games
- Filtering duplicate game titles
- Returning the final recommendations

This module acts as the primary connection between the precomputed machine learning artifacts and the Streamlit interface.

### `src/data_preprocessing.py`

Contains the data cleaning and feature engineering pipeline.

Its responsibilities include:

- Downloading or locating the Steam dataset
- Loading the raw CSV
- Removing unusable records
- Handling missing values
- Parsing structured dataset fields
- Extracting Steam tag names
- Normalizing genres, tags, and developers
- Constructing combined metadata features
- Preparing developer features
- Preparing the dataset for TF-IDF vectorization

The preprocessing pipeline transforms the original Steam data into the structured features required by the recommendation model.

### `src/build_model.py`

Provides the offline model-building pipeline.

Instead of rebuilding the recommendation model whenever the web application starts, this script performs the expensive preparation work ahead of time.

Running:

```bash
python src/build_model.py
```

performs the following workflow:

```text
Prepare Steam Dataset
        │
        ▼
Create Model Features
        │
        ▼
Build TF-IDF Models
        │
        ├── Metadata
        ├── Description
        └── Developer
        │
        ▼
Generate Sparse Matrices
        │
        ▼
Create Optimized Runtime Dataset
        │
        ▼
Save Model Artifacts
```

The resulting files are then loaded directly by the deployed application.

### `data/raw/`

Contains the original Steam dataset used during model development and artifact generation.

```text
games_march2025_full.csv
```

Because the raw dataset is large and can be retrieved separately, it is excluded from Git version control.

### `data/processed/`

Contains the optimized game catalog used by the running application.

```text
games_processed.pkl
```

Only the columns required during application runtime are retained, significantly reducing the size of the processed dataset.

### `models/`

Contains the precomputed machine learning artifacts generated by `build_model.py`.

Three separate feature spaces are stored:

```text
Metadata
├── metadata_vectorizer.pkl
└── metadata_matrix.npz

Description
├── description_vectorizer.pkl
└── description_matrix.npz

Developer
├── developer_vectorizer.pkl
└── developer_matrix.npz
```

The `.pkl` files contain the fitted TF-IDF vectorizers, while the `.npz` files contain the corresponding sparse feature matrices.

### `requirements.txt`

Defines the Python dependencies required to run GameMatch.

Major dependencies include:

```text
pandas
numpy
scikit-learn
scipy
rapidfuzz
streamlit
streamlit-searchbox
kagglehub
```

This file allows the application's Python environment to be recreated locally or during deployment.

### `.gitignore`

Prevents unnecessary development files and large source data from being committed to the repository.

Examples include:

```text
.venv/
__pycache__/
*.pyc
data/raw/
```

The raw Steam dataset and local Python environment therefore remain outside version control while the source code and required deployment resources remain reproducible.

## 🚀 Installation & Running Locally

GameMatch can be run locally using Python and Streamlit.

### Prerequisites

Before installing GameMatch, make sure the following are installed:

- **Python 3**
- **Git**
- **pip**

A Python virtual environment is recommended to keep project dependencies isolated.

### 1. Clone the Repository

Clone the GameMatch repository from GitHub:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Navigate into the project directory:

```bash
cd GameMatch
```

### 2. Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv .venv
```

#### Windows PowerShell

Activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents script execution, the environment can be activated for the current session using:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The primary dependencies include:

```text
pandas
numpy
scikit-learn
scipy
rapidfuzz
streamlit
streamlit-searchbox
kagglehub
```

### 4. Model Artifacts

GameMatch uses precomputed model artifacts so that the full dataset does not need to be processed every time the application starts.

The application expects the following files:

```text
data/
└── processed/
    └── games_processed.pkl

models/
├── metadata_matrix.npz
├── metadata_vectorizer.pkl
├── description_matrix.npz
├── description_vectorizer.pkl
├── developer_matrix.npz
└── developer_vectorizer.pkl
```

If these artifacts are already included with the project, the application can be launched immediately after installing the dependencies.

### 5. Rebuilding the Model

The recommendation artifacts can also be regenerated from the source Steam dataset.

Run:

```bash
python src/build_model.py
```

The build pipeline will:

1. Retrieve/load the Steam dataset.
2. Clean and preprocess the game data.
3. Parse genres, tags, developers, and other required fields.
4. Generate the model features.
5. Fit the metadata TF-IDF vectorizer.
6. Fit the description TF-IDF vectorizer.
7. Fit the developer TF-IDF vectorizer.
8. Generate the corresponding sparse matrices.
9. Create the optimized runtime game dataset.
10. Save the generated artifacts to `data/processed/` and `models/`.

Once this process completes, GameMatch is ready to run.

### 6. Launch GameMatch

Start the Streamlit application:

```bash
streamlit run src/app.py
```

Streamlit will start a local development server and provide an address similar to:

```text
Local URL: http://localhost:8501
```

Open the provided address in a web browser to use GameMatch.

### 7. Using the Application

Once GameMatch is running:

1. Search for a game using the autocomplete search box.
2. Select a game you enjoy.
3. Add up to **five games** to your preference profile.
4. Select **Find Recommendations**.
5. GameMatch will compare your preferences against the Steam game catalog.
6. Browse the generated recommendations.
7. Click a game's header image to open its Steam store page.

The selected games can be removed individually or cleared to create a new preference profile.

### Development Workflow

When modifying only the Streamlit interface or recommendation logic, the model artifacts generally do not need to be rebuilt.

The artifacts should be regenerated when changes are made to areas such as:

```text
Dataset
Feature engineering
Text preprocessing
TF-IDF configuration
Model feature spaces
```

After making changes that affect model generation, run:

```bash
python src/build_model.py
```

Then restart the application:

```bash
streamlit run src/app.py
```

## ☁️ Deployment

GameMatch is designed for deployment as a **Streamlit web application** using **Streamlit Community Cloud** and GitHub.

The deployment architecture takes advantage of GameMatch's offline model-building pipeline so that the hosted application does not need to download and process the complete Steam dataset whenever it starts.

### Deployment Architecture

```text
                 DEVELOPMENT / BUILD
                        │
                        ▼
               Steam Games Dataset
                  (~95,000 games)
                        │
                        ▼
                 build_model.py
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
    Processed Game Data       TF-IDF Artifacts
            │                       │
            └───────────┬───────────┘
                        │
                        ▼
                  GitHub Repository
                        │
                        ▼
              Streamlit Community Cloud
                        │
                        ▼
                   src/app.py
                        │
                        ▼
               Load Model Artifacts
                        │
                        ▼
              GameMatch Web Application
```

### Precomputed Deployment Artifacts

GameMatch performs model preparation before deployment rather than rebuilding the recommendation system when the hosted application starts.

The deployed application loads:

```text
data/processed/games_processed.pkl

models/metadata_matrix.npz
models/metadata_vectorizer.pkl

models/description_matrix.npz
models/description_vectorizer.pkl

models/developer_matrix.npz
models/developer_vectorizer.pkl
```

Together, these files contain the processed game catalog and the precomputed feature representations required by the recommendation engine.

This approach keeps the deployment process lightweight compared with rebuilding the complete model from the raw Steam dataset every time the application environment starts.

### Application Startup

When the deployed application starts, Streamlit executes:

```text
src/app.py
```

The application then initializes the recommendation engine and loads the processed dataset and model artifacts.

Streamlit resource caching is used so these resources do not need to be repeatedly loaded during normal application reruns.

The runtime workflow is therefore:

```text
Application Starts
       │
       ▼
Load Processed Game Catalog
       │
       ▼
Load Sparse TF-IDF Matrices
       │
       ▼
Cache Application Resources
       │
       ▼
GameMatch Ready
       │
       ▼
User Selects Games
       │
       ▼
Generate Recommendations On Demand
```

### GitHub Integration

The GameMatch source code is maintained using Git and hosted on GitHub.

The repository contains the source code, Python dependencies, processed runtime data, and model artifacts required by the deployed application.

Large raw source data is excluded from version control:

```text
data/raw/
```

The raw Steam dataset is therefore used during local model generation but is not required for normal application runtime.

### Dependency Management

Python dependencies are defined in:

```text
requirements.txt
```

During deployment, the application environment installs these dependencies before launching GameMatch.

This ensures that the hosted application has access to the same major libraries used during development, including:

```text
Streamlit
pandas
NumPy
scikit-learn
SciPy
RapidFuzz
streamlit-searchbox
```

### Deployment Workflow

The production deployment workflow follows:

```text
Develop / Test Locally
        │
        ▼
Rebuild Model Artifacts (if required)
        │
        ▼
Test GameMatch Locally
        │
        ▼
Commit Changes with Git
        │
        ▼
Push to GitHub
        │
        ▼
Streamlit Cloud Deployment
        │
        ▼
Test Production Application
```

Changes to the user interface or recommendation logic can generally be deployed directly after testing.

Changes to preprocessing, feature engineering, or TF-IDF configuration require the model artifacts to be regenerated before deployment.

### Live Application

Once deployment is complete, the hosted version of GameMatch will be available here:

```text
Live Demo:  https://game-match.streamlit.app/
```

The application can also be run locally by following the installation instructions above.

## ⚠️ Limitations & Future Improvements

GameMatch is designed as a lightweight, content-based recommendation system. While the current model produces relevant recommendations using Steam metadata, several limitations provide opportunities for future development.

### Current Limitations

#### Content-Based Recommendations Only

GameMatch currently makes recommendations entirely from information describing each game.

The model considers:

```text
Genres
Community Tags
Descriptions
Developers
```

It does not currently incorporate behavioral information such as:

- User ratings
- Playtime
- Purchases
- Wishlists
- Recommendation clicks
- Similar users' preferences

As a result, GameMatch primarily answers:

> "Which games are most similar to the games this user already enjoys?"

rather than:

> "Which games are users with similar tastes most likely to enjoy?"

---

#### Fixed Feature Weights

The current recommendation model uses manually configured weights:

```text
Genres & Tags  → 65%
Description    → 25%
Developer      → 10%
```

These weights provide a practical balance between the three feature spaces, but they are not currently learned from user behavior.

Future versions could use recommendation feedback or evaluation data to optimize these weights automatically.

---

#### Metadata Quality

Recommendation quality depends on the quality and completeness of the underlying Steam dataset.

Some games may have:

- Missing community tags
- Missing descriptions
- Limited metadata
- Inconsistent developer information

GameMatch preserves games with partially missing information whenever possible, but incomplete metadata can reduce the amount of information available for similarity calculations.

---

#### TF-IDF Semantic Limitations

TF-IDF is effective for identifying shared terms but does not deeply understand the semantic meaning of natural language.

For example, two descriptions could describe similar gameplay concepts using very different terminology and receive less similarity than expected.

More advanced language representations could potentially capture these relationships more effectively.

---

#### No Persistent User Profiles

GameMatch currently creates a preference profile from the games selected during the current session.

The application does not currently maintain persistent:

```text
User accounts
Favorite games
Recommendation history
Ratings
Preference profiles
```

Each new selection therefore generates a new temporary preference profile.

---

#### Similarity Does Not Equal Preference Probability

The internal weighted similarity score represents how closely a game matches the generated preference profile.

It should not be interpreted as the probability that a user will enjoy a game.

For this reason, GameMatch uses similarity primarily for **ranking recommendations** rather than presenting the value as a confidence percentage.

---

### Future Improvements

GameMatch's modular architecture allows additional recommendation techniques and application features to be incorporated in future versions.

#### Hybrid Recommendation System

A future version could combine the existing content-based model with collaborative filtering.

```text
Content-Based Model
        │
        ├─────────────┐
        │             │
        ▼             ▼
Game Metadata     User Behavior
                      │
                      ▼
              Collaborative Model
                      │
        ┌─────────────┘
        ▼
 Hybrid Recommendation Engine
        │
        ▼
 Personalized Recommendations
```

This would allow GameMatch to consider both **what games are similar** and **what players with similar preferences enjoy**.

---

#### User Feedback

Users could provide feedback on recommendations using controls such as:

```text
👍 Interested
👎 Not Interested
⭐ Favorite
```

This feedback could be stored and used to improve future recommendations.

Over time, GameMatch could learn more detailed preference profiles rather than relying exclusively on the games initially selected by the user.

---

#### Learned Feature Weights

Instead of manually assigning:

```text
65% Metadata
25% Description
10% Developer
```

future versions could learn the relative importance of each feature space from user interactions or labeled recommendation data.

This would allow the weighting system to adapt based on observed recommendation quality.

---

#### Semantic Embeddings

The description model could eventually be expanded beyond TF-IDF using semantic text embeddings.

A future architecture could compare:

```text
Current:

Description
    │
    ▼
 TF-IDF
    │
    ▼
Similarity


Potential Future Version:

Description
    │
    ▼
Text Embedding Model
    │
    ▼
Semantic Vector
    │
    ▼
Similarity
```

This could improve GameMatch's ability to recognize games that describe similar concepts using different vocabulary.

---

#### User Accounts and Persistent Profiles

Future versions could introduce persistent user accounts that store:

- Favorite games
- Previous recommendations
- Likes and dislikes
- Recommendation history
- Custom preference profiles

This would allow recommendations to improve over multiple sessions.

---

#### Recommendation Explanations

GameMatch could provide explanations describing why a particular title was recommended.

For example:

```text
Recommended because you enjoy:

• Open-world RPGs
• Dark fantasy
• Souls-like combat
• Exploration-focused games
```

Providing these explanations would make the recommendation system more transparent and help users understand why specific games appear in their results.

---

#### Advanced Filtering

Future versions could allow users to filter recommendations by additional Steam metadata such as:

```text
Price
Release date
Genre
Single-player / Multiplayer
Controller support
Operating system
```

This would allow users to combine similarity-based recommendations with practical purchasing preferences.

---

### Development Direction

The current version of GameMatch focuses on providing a fast and understandable **content-based recommendation engine**.

Future development can progressively expand the system:

```text
V1
Content-Based Recommendations
        │
        ▼
V2
User Feedback + Persistent Preferences
        │
        ▼
V3
Hybrid Content + Collaborative Filtering
        │
        ▼
V4
Learned Personalization + Semantic Models
```

This provides a path for GameMatch to evolve from a content-similarity application into a more comprehensive personalized recommendation platform.

## 📈 Challenges & Engineering Decisions

Developing GameMatch involved several challenges related to recommendation quality, application performance, memory usage, search responsiveness, and deployment. The architecture evolved throughout development as these issues were identified and addressed.

### Scaling Recommendations Across ~95,000 Games

One of the first architectural considerations was how to calculate recommendations efficiently across approximately 95,000 games.

A straightforward approach would be to precompute the similarity between every pair of games. However, this would require approximately:

```text
95,000 × 95,000
≈ 9 billion similarity values
```

Storing and calculating this matrix would be unnecessary for GameMatch's use case.

**Solution:** GameMatch calculates similarity on demand.

When recommendations are requested, the selected games are converted into a user preference profile and that single profile is compared against the catalog.

```text
Instead of:

Game × Every Other Game

GameMatch uses:

User Preference Profile × Game Catalog
```

This significantly reduces the amount of computation and storage required.

---

### Improving Recommendation Quality

The original recommendation model primarily relied on genres and Steam community tags.

While these features produced useful recommendations, games with nearly identical metadata could receive extremely similar scores even when other characteristics differed.

**Solution:** The recommendation engine was expanded into three independent feature spaces:

```text
Genres + Tags
      │
      ▼
Metadata TF-IDF


Short Description
      │
      ▼
Description TF-IDF


Developer
      │
      ▼
Developer TF-IDF
```

The resulting similarities are combined using weighted scoring:

```text
65% Metadata
25% Description
10% Developer
```

Separating these features also makes their influence independently configurable.

---

### Supporting Multiple Favorite Games

Generating recommendations from a single game can make results overly dependent on the characteristics of that title.

GameMatch needed a way to represent the user's broader preferences when several favorite games were provided.

**Solution:** GameMatch averages the TF-IDF vectors of the selected games within each feature space.

```text
Game A ─┐
Game B ─┼──► Average Vector ──► Preference Profile
Game C ─┘
```

This allows up to five selected games to contribute to a combined representation of the user's interests.

---

### Searching a Catalog of ~95,000 Titles

An early interface approach used a standard selection component containing the complete game catalog.

Loading approximately 95,000 titles into a single dropdown created unnecessary interface overhead and reduced responsiveness.

**Solution:** The search interface was redesigned as a dynamic autocomplete system.

```text
User Types Query
       │
       ▼
Substring Matching
       │
       ▼
Fuzzy Matching
       │
       ▼
Return Top Results
```

Only a small number of relevant results are displayed.

RapidFuzz was also incorporated so searches can tolerate partial titles and minor spelling mistakes.

---

### Reducing Application Startup Time

Earlier versions of GameMatch performed dataset preprocessing and TF-IDF model generation as part of application initialization.

This meant expensive operations could be repeated even though the underlying model had not changed.

**Solution:** Model generation was separated from application runtime.

```text
build_model.py
      │
      ▼
Generate Artifacts Once
      │
      ▼
Save to Disk
      │
      ▼
Application Loads Artifacts
```

The application now loads precomputed data and sparse matrices instead of rebuilding them during normal startup.

Streamlit resource caching further prevents these resources from being repeatedly loaded during application reruns.

---

### Reducing the Processed Dataset Size

The initial processed game dataset retained many columns from the original Steam dataset and occupied approximately:

```text
487 MB
```

Most of these fields were required only during preprocessing and model generation.

They did not need to be loaded by the deployed application.

**Solution:** The runtime dataset was reduced to:

```text
appid
name
genres
developers
short_description
header_image
```

This reduced the processed artifact to approximately:

```text
35 MB
```

representing a reduction of more than **90%** while maintaining the information required by the application.

---

### Handling Duplicate Steam Titles

The Steam dataset can contain multiple records sharing the same displayed game title.

Because each record is independently ranked by the recommendation engine, duplicate titles could occasionally appear in the final recommendations.

Simply removing duplicates after selecting the top results could result in fewer recommendations being displayed.

**Solution:** GameMatch filters duplicates while traversing the ranked candidates.

```text
Ranked Results
      │
      ▼
Already Selected?
   Yes → Skip
      │
      No
      ▼
Title Already Seen?
   Yes → Skip
      │
      No
      ▼
Add Recommendation
```

The process continues until the requested number of unique recommendations has been collected.

---

### Keeping Search Flexible Without Sacrificing Performance

Exact title matching alone would require users to know the correct spelling and formatting of every game.

However, performing expensive fuzzy matching unnecessarily could also reduce search responsiveness.

**Solution:** GameMatch combines two approaches.

First, it searches for inexpensive substring matches:

```text
"elden"
   ↓
"Elden Ring"
```

Fuzzy matching is then used to supplement those results when needed:

```text
"eldin rng"
    ↓
RapidFuzz
    ↓
"Elden Ring"
```

This provides typo tolerance while keeping the search experience responsive.

---

### Separating Build-Time and Runtime Responsibilities

A major architectural decision was determining which operations belong in the model-building pipeline and which belong in the deployed application.

GameMatch now maintains a clear separation:

| Build Time | Runtime |
|---|---|
| Download raw dataset | Load processed dataset |
| Clean and parse data | Search games |
| Engineer features | Manage selections |
| Fit TF-IDF models | Build preference profiles |
| Generate sparse matrices | Calculate cosine similarity |
| Create deployment dataset | Rank recommendations |
| Save model artifacts | Display results |

This separation makes the application easier to maintain, faster to initialize, and more practical to deploy.

---

### Engineering Approach

Many of GameMatch's current design decisions resulted from identifying bottlenecks during development and iteratively improving the architecture.

The project evolved from a relatively simple content-similarity prototype into a system with:

```text
Multiple Feature Spaces
        +
Multi-Game Preference Profiles
        +
Sparse Matrix Storage
        +
Precomputed Model Artifacts
        +
Dynamic Fuzzy Search
        +
Runtime Resource Caching
        +
Deployment-Optimized Data
```

This iterative approach allowed recommendation quality, performance, usability, and deployment requirements to be addressed independently while keeping the overall architecture understandable.

## 📚 What I Learned

Building GameMatch provided experience across the full lifecycle of a machine learning application, from working with raw data and designing a recommendation algorithm to optimizing the application and preparing it for deployment.

The project began as a relatively simple idea—recommend games based on titles a user already enjoys—but developing it into a usable application required solving problems across **machine learning, data processing, software engineering, performance optimization, and user interface design**.

### Building a Recommendation System

GameMatch provided hands-on experience designing a **content-based recommendation system** from the ground up.

This included learning how to:

- Represent games using meaningful features
- Convert text and categorical metadata into numerical vectors
- Compare high-dimensional vectors using cosine similarity
- Combine multiple similarity signals
- Create preference profiles from multiple user-selected games
- Rank and filter recommendation candidates

One of the most important lessons was that recommendation quality depends heavily on **how the underlying information is represented**, not simply on the similarity algorithm itself.

The model evolved from primarily comparing genres and tags to using three independent feature spaces:

```text
Metadata
Description
Developer
```

This made the recommendation system more flexible and allowed different types of information to contribute independently to the final ranking.

### Feature Engineering

The project reinforced the importance of preparing data specifically for the problem being solved.

Raw Steam metadata could not simply be passed directly into the recommendation model.

Features needed to be parsed, cleaned, normalized, and transformed into useful representations.

For example:

```text
"Dark Fantasy" → "dark_fantasy"
"Open World"   → "open_world"
"Souls-like"   → "souls_like"
```

Preserving these concepts as individual features helped maintain their meaning during vectorization.

The project demonstrated how feature engineering decisions can directly influence the behavior and quality of a machine learning system.

### Working with Sparse Data

Using TF-IDF across approximately 95,000 games created large, high-dimensional feature spaces.

However, most games only contain a small fraction of all possible terms.

This provided practical experience working with **sparse matrices** and understanding why they are useful for machine learning applications involving text.

Instead of representing every possible feature value in dense arrays, GameMatch stores only meaningful non-zero values using SciPy sparse matrices.

This significantly reduces unnecessary memory and storage requirements.

### Thinking About Computational Complexity

GameMatch also demonstrated the importance of considering how an algorithm scales as a dataset grows.

Precomputing similarity between approximately 95,000 games would require roughly:

```text
95,000 × 95,000
≈ 9 billion comparisons
```

Instead of creating an all-pairs similarity matrix, GameMatch calculates similarity between a generated user preference profile and the catalog only when recommendations are requested.

This reinforced an important software engineering principle:

> A solution that works on a small dataset is not necessarily the right solution when the system scales.

### Separating Model Building from Application Runtime

One of the largest architectural improvements was separating expensive model preparation from the running web application.

The final architecture distinguishes between:

```text
BUILD TIME
──────────
Data preprocessing
Feature engineering
TF-IDF fitting
Sparse matrix generation
Artifact creation


RUNTIME
───────
Load artifacts
Search games
Build user profiles
Calculate similarity
Rank results
Display recommendations
```

This made the application faster to start and easier to deploy.

It also provided practical experience with a common machine learning engineering pattern: **perform expensive preparation ahead of time and load the resulting artifacts during inference**.

### Optimizing Data for Deployment

Preparing GameMatch for deployment highlighted the difference between development data and production data.

The original processed dataset occupied approximately:

```text
487 MB
```

After analyzing which fields were actually required by the running application, the runtime dataset was reduced to approximately:

```text
35 MB
```

without changing the recommendation model.

This reinforced the importance of examining what data an application actually needs rather than automatically carrying the entire development dataset into production.

### Designing Search for Large Datasets

Working with approximately 95,000 game titles also introduced user-interface performance considerations.

A standard dropdown containing the entire catalog was technically functional but produced a poor search experience.

Replacing it with dynamic autocomplete and fuzzy matching demonstrated how backend data size can directly influence frontend design.

The final search system combines:

```text
Substring Matching
        +
Fuzzy Matching
        +
Limited Search Results
        =
Responsive Autocomplete
```

This provided experience balancing usability with computational performance.

### Building Beyond the Machine Learning Model

One of the biggest lessons from GameMatch was that creating a machine learning model is only one part of building a complete application.

A useful ML application also requires:

```text
Data Pipeline
     +
Machine Learning Model
     +
Application Logic
     +
Search
     +
User Interface
     +
Performance Optimization
     +
Version Control
     +
Deployment
```

The recommendation algorithm may be the core of GameMatch, but making the system usable required integrating all of these components.

### Iterative Software Development

GameMatch was developed iteratively rather than attempting to design the final architecture immediately.

Individual problems were identified and improved as the application evolved:

```text
Basic Recommendation Prototype
            │
            ▼
Multi-Feature Recommendation Model
            │
            ▼
Multi-Game Preference Profiles
            │
            ▼
Fuzzy Autocomplete Search
            │
            ▼
Precomputed Model Artifacts
            │
            ▼
Performance Optimization
            │
            ▼
Deployment-Ready Application
```

This process reinforced the value of building a working version first, identifying real limitations, and then improving the architecture based on observed problems.

### Overall Takeaway

GameMatch provided practical experience taking a machine learning concept beyond an experimental model and developing it into a complete software application.

The project strengthened skills in:

- **Python software development**
- **Machine learning and recommendation systems**
- **TF-IDF and text feature representation**
- **Feature engineering**
- **Cosine similarity**
- **pandas and NumPy**
- **Sparse matrix operations**
- **Data preprocessing**
- **Performance optimization**
- **Application architecture**
- **Streamlit development**
- **Git and GitHub**
- **Deployment preparation**

Most importantly, the project demonstrated how machine learning, data engineering, and software engineering decisions must work together to turn an algorithm into a usable application.

## 🙏 Acknowledgments & Data Source

GameMatch was developed using the **Steam Games Dataset 2025** created and published by **Artem Miloff** on Kaggle.

The dataset provides metadata for approximately 95,000 Steam games, including game titles, genres, community tags, developers, descriptions, Steam App IDs, header images, and other information used throughout this project.

### Dataset

**Steam Games Dataset 2025 — Artem Miloff**

[View the dataset on Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)

The dataset served as the primary data source for GameMatch's preprocessing pipeline, feature engineering, TF-IDF representations, and recommendation system.

All credit for the collection and publication of the original Steam dataset belongs to the dataset creator. GameMatch independently processes and transforms this data to build its recommendation engine.

### Technologies & Open-Source Libraries

GameMatch was also made possible through the open-source Python ecosystem, including:

- pandas
- NumPy
- scikit-learn
- SciPy
- RapidFuzz
- Streamlit
- streamlit-searchbox
- KaggleHub

These libraries provide the data processing, machine learning, scientific computing, search, and web application functionality used throughout the project.


## Author

## Dylan Santiago