import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.decomposition import TruncatedSVD

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

# For each user, hold out 1 rating for test (preferably a positive, rating >= 4)

def leave_one_out_split(df, positive_threshold=4.0, seed=42):
    rng = np.random.default_rng(seed)
    test_rows = []
    keep_mask = np.ones(len(df), dtype=bool)

    # group by user
    for uid, grp in df.groupby('userId'):
        pos_idx = grp.index[grp['rating'] >= positive_threshold].tolist()
        if len(pos_idx) > 0:
            chosen = rng.choice(pos_idx, size=1)[0]
        else: 
            chosen = rng.choice(grp.index.values, size=1)[0]
        test_rows.append(chosen)
        keep_mask[df.index.get_loc(chosen)] = False
    
    test_df = df.loc[test_rows]
    train_df = df[keep_mask]
    return train_df, test_df

train_ratings, test_ratings = leave_one_out_split(ratings, positive_threshold=4.0, seed=42)

print("Train size:", train_ratings.shape, "Test size:", test_ratings.shape)

# Build pivot on train set only
user_item = train_ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Keep quick maps
all_users = user_item.index.to_numpy()
all_items = user_item.columns.to_numpy()

# Compute user mean ratings, used for mean centering and a safe fallback
user_means = user_item.mean(axis=1)

# Mean center
user_item_centered = user_item.sub(user_means, axis=0).fillna(0.0)

sim_users = cosine_similarity(user_item_centered.values)

user_sim_df = pd.DataFrame(sim_users, index=all_users, columns=all_users)

print("Similarity shape:", user_sim_df.shape)

def recommend_for_user(target_user, k_neighbors=40, top_n=10):
    if target_user not in user_item.index:
        raise ValueError("User not found in training data")
    
    # Get the target row and similarities to all users
    sims = user_sim_df.loc[target_user]

    sims = sims.drop(target_user)

    # Get top K neighbors by similarity
    top_neighbors = sims.nlargest(k_neighbors).index

    neighbor_centered = user_item_centered.loc[top_neighbors]

    # Similarities vector aligned to neighbor rows
    neighbor_sims = sims.loc[top_neighbors].to_numpy().reshape(-1, 1)

    # We only want items the target user has not rated in train
    seen_items = user_item.loc[target_user].dropna().index
    candidate_items = [i for i in user_item.columns if i not in seen_items]

    # Compute predicted scores for each candidate
    preds = {}
    for item in candidate_items:
        residuals = neighbor_centered[item].to_numpy().reshape(-1, 1)
        mask = residuals.flatten() != 0.0
        if not np.any(mask):
            continue
        sims_used = neighbor_sims[mask]
        res_used = residuals[mask]
        denom = np.sum(np.abs(sims_used))
        if denom == 0.0:
            continue
        delta = np.sum(sims_used * res_used) / denom
        preds[item] = user_means.loc[target_user] + delta

    # Top N by predicted score
    if len(preds) == 0:
        return pd.DataFrame(columns=["movieId", "pred_score", "title"])
    
    recs = (
        pd.DataFrame([(iid, s) for iid, s in preds.items()], columns=["movieId","pred_score"])
        .sort_values("pred_score", ascending=False)
        .head(top_n)
        .merge(movies[["movieId","title"]], on="movieId", how="left")
    )
    return recs

# Example, pick a user from the test set
example_user = int(test_ratings['userId'].sample(1, random_state=7).iloc[0])
recommend_for_user(example_user, k_neighbors=40, top_n=10)

def precision_at_k(user_item, user_sim_df, train_ratings, test_ratings, k_neighbors=40, K=10, positive_threshold=4.0):
    test_pos = (
        test_ratings[test_ratings['rating'] >= positive_threshold]
        .groupby('userId')['movieId']
        .apply(set)
        .to_dict()
    )

    users = list(test_ratings['userId'].unique())
    precisions = []

    for u in users:
        if u not in user_item.index:
            continue

        recs = recommend_for_user(u, k_neighbors=k_neighbors, top_n=K)
        if recs.empty:
            precisions.append(0.0)
            continue

        topk_items = set(recs['movieId'].tolist())
        pos_items = test_pos.get(u, set())

        hits = len(topk_items.intersection(pos_items))
        precisions.append(hits / K)

    return float(np.mean(precisions)) if len(precisions) > 0 else 0.0

p_at_10 = precision_at_k(user_item, user_sim_df, train_ratings, test_ratings, k_neighbors=40, K=10, positive_threshold=4.0)
print("Precision@10:", round(p_at_10, 4))

def show_user_history(user_id, n=10):
    hist = (
        train_ratings[train_ratings['userId'] == user_id]
        .merge(movies[['movieId', 'title']], on='movieId', how='left')
        .sort_values('rating', ascending=False)
        .head(n)
    )
    return hist[['movieId', 'title', 'rating']]

print(show_user_history(example_user))
print(recommend_for_user(example_user, k_neighbors=40, top_n=10))

# Bonus 1) Item based collaborative filtering
item_user = train_ratings.pivot_table(index='movieId', columns='userId', values='rating')

# Item mean centering
item_means = item_user.mean(axis=1)
item_user_centered = item_user.sub(item_means, axis=0).fillna(0.0)

# Item item cosine similarity
sim_items = cosine_similarity(item_user_centered.values)
item_sim_df = pd.DataFrame(sim_items, index=item_user.index, columns=item_user.index)

def recommend_for_user_item_based(target_user, k_neighbors=40, top_n=10):
    if target_user not in user_item.index:
        raise ValueError("User not found in training data")
    
    # Items the user has rated in train
    user_ratings = user_item.loc[target_user].dropna()
    seen_items = set(user_ratings.index)

    scores = {}
    for cand in item_user.index:
        if cand in seen_items:
            continue
        # Similarities to items the user has been seen
        sims = item_sim_df.loc[cand, list(seen_items)]
        residuals = user_ratings.loc[list(seen_items)] - item_means.loc[list(seen_items)]

        # select top k similar seen items to reduce noise
        topk_idx = sims.nlargest(k_neighbors).index
        sims_k = sims.loc[topk_idx].to_numpy()
        res_k = residuals.loc[topk_idx].to_numpy()

        denom = np.sum(np.abs(sims_k))
        if denom == 0.0:
            continue
        delta = np.sum(sims_k * res_k) / denom
        scores[cand] = item_means.loc[cand] + delta
    
    if len(scores) == 0:
        return pd.DataFrame(columns=["movieId","pred_score","title"])
    
    recs = (
        pd.DataFrame([(iid, s) for iid, s in scores.items()], columns=["movieId","pred_score"])
        .sort_values("pred_score", ascending=False)
        .head(top_n)
        .merge(movies[["movieId","title"]], on="movieId", how="left")
    )
    return recs

# Bonus 2) Matrix factorization with TruncatedSVD
# Build user item matrix on train and fill missing with 0 after mean centering
# We will center by user to reduce bias
user_item_train = train_ratings.pivot_table(index='userId', columns='movieId', values='rating')
user_means_train = user_item_train.mean(axis=1)
M_centered = user_item_train.sub(user_means_train, axis=0).fillna(0.0).values

# Factorize to r latent factors
r = 50
svd = TruncatedSVD(n_components=r, random_state=42)
U = svd.fit_transform(M_centered)           # users by r
SigmaVT = svd.components_                   # r by items

# Reconstruct scores, then add user means back
score_matrix = U @ SigmaVT                  # centered predictions
score_matrix = score_matrix + user_means_train.to_numpy().reshape(-1,1)

svd_scores = pd.DataFrame(score_matrix, index=user_item_train.index, columns=user_item_train.columns)

def recommend_for_user_svd(user_id, top_n=10):
    if user_id not in svd_scores.index:
        raise ValueError("User not found in training data")
    user_scores = svd_scores.loc[user_id]
    seen = user_item_train.loc[user_id].dropna().index
    candidates = user_scores.drop(index=seen).sort_values(ascending=False).head(top_n)
    return (
        candidates.reset_index().rename(columns={"index":"movieId", user_id:"pred"})
        .merge(movies[["movieId","title"]], on="movieId", how="left")
        .rename(columns={0:"pred_score"})
    )