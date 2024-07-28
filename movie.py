import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load Data
@st.cache_resource
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

# Create User-Item Matrix
@st.cache_resource
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]
    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# Define Similarity Function
def find_similar_movies(_X, movie_id, k, metric='cosine', show_distance=False):
    neighbour_ids = []
    movie_ind = movie_mapper.get(movie_id)
    if movie_ind is None:
        st.write(f"Movie ID {movie_id} not found.")
        return neighbour_ids
    
    movie_vec = _X[movie_ind]
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(_X)
    
    movie_vec = movie_vec.reshape(1, -1)
    
    if show_distance:
        distances, indices = kNN.kneighbors(movie_vec, return_distance=True)
    else:
        indices = kNN.kneighbors(movie_vec, return_distance=False)
    
    # Get the indices of the nearest neighbors, skipping the first one (which is the movie itself)
    neighbour_ids = [movie_inv_mapper[idx] for idx in indices[0][1:]]
    
    return neighbour_ids

# Streamlit App
st.title("Movie Recommendation System")

# Ask if user wants to apply filters
apply_filters = st.checkbox("Apply filters for year and rating")

if apply_filters:
    year_input = st.number_input("Enter the year:", min_value=1900, max_value=2100, value=2000, step=1)
    min_rating = st.slider("Minimum average rating:", min_value=1, max_value=5, value=3, step=1)

    # Filter movies based on the year and minimum rating
    filtered_movies = movies[movies['title'].str.contains(str(year_input), case=False, na=False)]
    if not filtered_movies.empty:
        filtered_movie_ids = filtered_movies['movieId'].unique()
        filtered_ratings = ratings[ratings['movieId'].isin(filtered_movie_ids) & (ratings['rating'] >= min_rating)]
        filtered_movie_ids = filtered_ratings['movieId'].unique()
        filtered_movies = filtered_movies[filtered_movies['movieId'].isin(filtered_movie_ids)]
    else:
        filtered_movies = pd.DataFrame(columns=['movieId', 'title'])  # Empty DataFrame if no movies match

    movie_titles = dict(zip(filtered_movies['movieId'], filtered_movies['title']))
else:
    # Show all movies if no filters are applied
    movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_options = list(movie_titles.values())

if not movie_options:
    st.write("No movies found.")
else:
    selected_movie_title = st.selectbox("Select a movie:", options=movie_options)

    if selected_movie_title:
        selected_movie_id = movies[movies['title'] == selected_movie_title]['movieId'].values[0]
        st.write(f"You selected: {selected_movie_title}")

        if apply_filters:
            # Filter ratings based on the selected movie and minimum rating
            relevant_ratings = ratings[(ratings['movieId'] == selected_movie_id) & (ratings['rating'] >= min_rating)]
            if relevant_ratings.empty:
                st.write("No ratings found for the selected movie with the specified rating.")
            else:
                similar_movie_ids = find_similar_movies(X, selected_movie_id, k=10)
                similar_movie_ids = [movie_id for movie_id in similar_movie_ids if movie_id in movie_titles]

                if similar_movie_ids:
                    st.write("You might also like:")
                    for movie_id in similar_movie_ids:
                        st.write(movie_titles.get(movie_id, "Movie not found"))
                else:
                    st.write("No similar movies found.")
        else:
            similar_movie_ids = find_similar_movies(X, selected_movie_id, k=10)
            similar_movie_ids = [movie_id for movie_id in similar_movie_ids if movie_id in movie_titles]

            if similar_movie_ids:
                st.write("You might also like:")
                for movie_id in similar_movie_ids:
                    st.write(movie_titles.get(movie_id, "Movie not found"))
            else:
                st.write("No similar movies found.")
