# Movie Recommendation System

This project implements a movie recommendation system using collaborative filtering and matrix factorization techniques on the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).

## Features

- **User-based collaborative filtering** (cosine similarity)
- **Item-based collaborative filtering**
- **Matrix factorization** (Truncated SVD)
- **Leave-one-out evaluation** for precision@K
- Filters users and movies with fewer than 20 ratings

## Project Structure

```
movie-recommendation-system/
├── data/
│   └── ml-100k/
│       ├── u.data
│       └── u.item
├── src/
│   └── model.py
└── README.md
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

Install dependencies:

```sh
pip install pandas numpy scikit-learn
```

## Usage

1. **Download the MovieLens 100K dataset** and place the files in `data/ml-100k/`.
2. **Run the model:**

   ```sh
   python src/model.py
   ```

   This will:
   - Load and preprocess the data
   - Filter users/movies with <20 ratings
   - Split ratings into train/test sets (leave-one-out)
   - Compute recommendations for a sample user
   - Print precision@10 for the recommender

## Main Algorithms

- `recommend_for_user(user_id, k_neighbors=40, top_n=10)`: User-based recommendations
- `recommend_for_user_item_based(user_id, k_neighbors=40, top_n=10)`: Item-based recommendations
- `recommend_for_user_svd(user_id, top_n=10)`: SVD-based recommendations
- `precision_at_k(...)`: Precision@K evaluation

## Evaluation

The system uses leave-one-out splitting, holding out one rating per user for testing (preferably a positive rating, ≥4). Precision@10 is computed to evaluate recommendation quality.