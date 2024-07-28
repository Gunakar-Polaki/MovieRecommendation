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

# Movie Selection
movie_titles = dict(zip(movies['movieId'], movies['title']))
movie_options = list(movie_titles.values())

# Add placeholder text for the selectbox
selected_movie_title = st.selectbox("Select a movie:", options=movie_options, index=None, format_func=lambda x: x if x else "Select a movie")

if selected_movie_title:
    selected_movie_id = movies[movies['title'] == selected_movie_title]['movieId'].values[0]
    st.write(f"You selected: {selected_movie_title}")
    similar_movie_ids = find_similar_movies(X, selected_movie_id, k=10)
    if similar_movie_ids:
        st.write("You might also like:")
        for movie_id in similar_movie_ids:
            st.write(movie_titles.get(movie_id, "Movie not found"))
    else:
        st.write("No similar movies found.")
else:
    st.write("Please select a movie to see recommendations.")
