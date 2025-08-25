import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score 
from sklearn.metrics.pairwise import cosine_similarity 

ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])
movies = pd.read_csv("data/ml-100k/u.item", sep="|", encoding="latin-1", names=["movieId","title","release_date","video_release_date","imdb_url",
                           "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
                           "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
                           "Mystery","Romance","Sci-Fi","Thriller","War","Western"])

print(ratings.head())
print(ratings.shape)
print(ratings['rating'].describe())
print(movies.head())

# Keep users with at least 20 ratings and movies with at least 20 ratings
min_user_ratings = 20
min_movie_ratings = 20

user_counts = ratings['userId'].value_counts()
movie_counts = ratings['movieId'].value_counts()

ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= min_user_ratings].index)]
ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_ratings].index)]

print("After filtering:", ratings.shape)