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