
"""
End-to-end pattern recognition pipeline for a movie ratings dataset.

This module provides utilities to:
- load and parse the raw dataset,
- filter users/items by rating frequency,
- build a user–item preference matrix,
- run clustering (custom Euclidean K-means, spherical/cosine K-means, Jaccard K-medoids),
- construct per-cluster neighbor-based latent training data with TruncatedSVD,
- train per-cluster MLP regressors in latent space and evaluate (RMSE/MAE), and
- generate histogram and 3D cluster visualizations saved under the `output/` directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Ensure the output directory exists at startup
if not os.path.isdir("output"):
    os.makedirs("output", exist_ok=True)

def load_npy_dataset(file_path):
    """
    Load a NumPy dataset from a .npy file.

    Args:
        file_path (str): Path to the .npy file

    Returns:
        numpy.ndarray: The loaded dataset
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

                           
        data = np.load(file_path)

        return data

    except Exception as e:
        print(f"Error loading NPY file: {e}")
        return None

def parse_ratings_to_dataframe(dataset):
    """
    Parse the movie ratings dataset into a pandas DataFrame.

    Args:
        dataset (numpy.ndarray): The loaded NPY dataset containing string records

    Returns:
        pandas.DataFrame: Parsed DataFrame with columns: user_id, movie_id, rating, date
    """
    try:
        parsed_data = []

        for record in dataset:
            try:
                # Split raw CSV-like line into fields (user_id, movie_id, rating, date...)
                parts = record.split(',')

                if len(parts) >= 4:
                    user_id = parts[0]
                    movie_id = parts[1]
                    rating = int(parts[2])

                    # Join the remainder as date string (handles commas inside date field)
                    date_str = ','.join(parts[3:])

                    try:
                        # Prefer strict day-month-year; otherwise fall back to pandas parser
                        if len(parts) == 4:
                            date_obj = datetime.strptime(date_str.strip(), "%d %B %Y")
                        else:
                            date_obj = pd.to_datetime(date_str.strip(), errors='coerce')

                        parsed_data.append({
                            'user_id': user_id,
                            'movie_id': movie_id,
                            'rating': rating,
                            'date': date_obj
                        })
                    except ValueError:
                        parsed_data.append({
                            'user_id': user_id,
                            'movie_id': movie_id,
                            'rating': rating,
                            'date': date_str.strip()
                        })

            except (ValueError, IndexError):
                # Skip malformed rows
                continue

        df = pd.DataFrame(parsed_data)

        return df

    except Exception as e:
        print(f"Error parsing dataset: {e}")
        return None

def find_unique_users_and_movies(df):
    """
    Find unique users (U) and unique movies (I) from the DataFrame.

    Args:
        df (pandas.DataFrame): The parsed ratings DataFrame

    Returns:
        tuple: (unique_users_set, unique_movies_set, num_users, num_movies)
    """
    unique_users = set(df['user_id'].unique())
    unique_movies = set(df['movie_id'].unique())

    return unique_users, unique_movies, len(unique_users), len(unique_movies)

def filter_users_and_movies(df, r_min, r_max):
    """
    Filter users and movies based on rating count constraints.

    Query 2 (Preprocessing): Restrict the sets of unique users U and unique items I
    to the corresponding sets Û and Î such that:

    R_min ≤ |φ(u)| ≤ R_max, ∀ u ∈ Û

    where |φ(u)| is the number of ratings made by user u.

    Args:
        df (pandas.DataFrame): The ratings DataFrame
        r_min (int): Minimum number of ratings per user
        r_max (int): Maximum number of ratings per user

    Returns:
        tuple: (filtered_df, filtered_users, filtered_movies, num_filtered_users, num_filtered_movies)
    """
    user_rating_counts = df.groupby('user_id').size()

    valid_users = user_rating_counts[(user_rating_counts >= r_min) & (user_rating_counts <= r_max)].index

    filtered_df = df[df['user_id'].isin(valid_users)]

    filtered_movies = set(filtered_df['movie_id'].unique())
    filtered_users = set(valid_users)

    return (filtered_df, filtered_users, filtered_movies,
            len(filtered_users), len(filtered_movies))

def plot_histograms(df, save_path=None):
    """
    Create and display histograms for:
    1. Number of ratings per user
    2. Time span of ratings per user

    Args:
        df (pandas.DataFrame): The ratings DataFrame
        save_path (str, optional): Path to save the plots. If None, displays plots.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    user_rating_counts = df.groupby('user_id').size()

    ax1.hist(user_rating_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Histogram: Number of Ratings per User', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Ratings', fontsize=12)
    ax1.set_ylabel('Number of Users', fontsize=12)
    ax1.grid(True, alpha=0.3)

    mean_ratings = user_rating_counts.mean()
    median_ratings = user_rating_counts.median()
    max_ratings = user_rating_counts.max()
    min_ratings = user_rating_counts.min()

    stats_text = '.1f'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    user_time_stats = df.groupby('user_id')['date'].agg(['min', 'max'])
    user_time_spans = (user_time_stats['max'] - user_time_stats['min']).dt.days

    user_time_spans = user_time_spans[user_time_spans > 0]

    ax2.hist(user_time_spans, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Histogram: Time Span of Ratings per User', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Span (days)', fontsize=12)
    ax2.set_ylabel('Number of Users', fontsize=12)
    ax2.grid(True, alpha=0.3)

    if len(user_time_spans) > 0:
        mean_span = user_time_spans.mean()
        median_span = user_time_spans.median()
        max_span = user_time_spans.max()
        min_span = user_time_spans.min()

        stats_text2 = '.1f'
        ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes,
                 verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histograms saved to: {os.path.abspath(save_path)}")
    else:
        plt.show()

    return fig, (ax1, ax2)

def build_preference_matrix(df, users, movies):
    """
    Build the user-item preference matrix R.

    Query 4: Create an alternative representation of the dataset as a set of preference vectors.
    Each row R_j represents the preference vector of user u_j over the filtered set of movies Î.

    Args:
        df (pandas.DataFrame): Filtered ratings DataFrame
        users (set): Filtered set of users (Û)
        movies (set): Filtered set of movies (Î)

    Returns:
        pandas.DataFrame: User-item preference matrix (rows=users, cols=movies)
                           Entry (j,k) = rating of user u_j for movie i_k, or 0 if no rating
    """
    print("Building preference matrix R...")

    users_sorted = sorted(users)
    movies_sorted = sorted(movies)

    print(f"Matrix dimensions: {len(users_sorted)} users × {len(movies_sorted)} movies")

    R_df = df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        fill_value=0,                               
        dropna=False                                         
    )

    R_df = R_df.reindex(index=users_sorted, columns=movies_sorted, fill_value=0)

    print("✓ Preference matrix R created successfully")
    print(f"Matrix shape: {R_df.shape}")
    print(f"Matrix sparsity: {(R_df == 0).sum().sum() / R_df.size * 100:.2f}% zeros")

    return R_df


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each row vector of a 2D array to unit L2 norm.

    Zero rows are kept as zero by dividing with a safe denominator of 1.0.

    Args:
        matrix (np.ndarray): Array of shape (n_samples, n_features).

    Returns:
        np.ndarray: Row-normalized array with the same shape.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms

def spherical_kmeans(X: np.ndarray, n_clusters: int, n_init: int = 10, max_iter: int = 100,
                     tol: float = 1e-4, random_state: int = 42):
    """
    Spherical K-Means clustering.
    - Rows are normalized to unit length (projected to the unit hypersphere)
    - Centroids are re-normalized to unit length each iteration
    - Objective maximizes total cosine similarity to centroids

    Returns:
        labels (np.ndarray), centroids (np.ndarray)
    """
    rng = np.random.default_rng(random_state)

    # Normalize rows to unit length so cosine similarity reduces to a dot product
    X_norm = _row_normalize(X.astype(float))

    best_objective = -np.inf
    best_labels = None
    best_centroids = None

    n_samples = X_norm.shape[0]

    for init_idx in range(n_init):
                                                                               
        # Prefer seeds from non-zero rows to avoid degenerate centroids
        non_zero_rows = np.where(np.linalg.norm(X_norm, axis=1) > 0)[0]
        if len(non_zero_rows) >= n_clusters:
            seed_indices = rng.choice(non_zero_rows, size=n_clusters, replace=False)
        else:
            seed_indices = rng.choice(np.arange(n_samples), size=n_clusters, replace=False)

        centroids = X_norm[seed_indices].copy()
        centroids = _row_normalize(centroids)

        for _ in range(max_iter):
                                                                                       
            # Cosine similarity for unit vectors is just the dot product
            similarities = X_norm @ centroids.T                           
            labels = np.argmax(similarities, axis=1)

                                                                      
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                members = X_norm[labels == k]
                if members.size == 0:
                    # Re-seed empty cluster to a random unit vector (from data or random)
                    ridx = rng.integers(0, n_samples)
                    candidate = X_norm[ridx]
                    if np.linalg.norm(candidate) == 0:
                                                       
                        candidate = rng.standard_normal(X_norm.shape[1])
                        candidate = candidate / (np.linalg.norm(candidate) + 1e-12)
                    new_centroids[k] = candidate
                else:
                    mean_vec = members.mean(axis=0)
                    norm = np.linalg.norm(mean_vec)
                    if norm == 0:
                        new_centroids[k] = centroids[k]
                    else:
                        new_centroids[k] = mean_vec / norm

            # Measure centroid movement to check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < tol:
                break

        # Objective: sum of each point's cosine similarity to its assigned centroid
        objective = np.sum(np.max(X_norm @ centroids.T, axis=1))
        if objective > best_objective:
            best_objective = objective
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids

                                                                   

class CustomKMeansMasked:
    """
    K-Means variant that uses the assignment's Euclidean distance:
      d(R_u, R_v) = sqrt(sum_k (R_u(k) - R_v(k))^2 * lambda_u(k) * lambda_v(k))
    Distances to centroids are computed only over coordinates rated by both
    the user and the centroid (centroid has a rating if at least one cluster member rated it).
    """
    def __init__(self, n_clusters: int, n_init: int = 5, max_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def _compute_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute masked centroids by averaging only over non-zero entries per feature.

        Args:
            X (np.ndarray): Data matrix of shape (n_samples, n_features).
            labels (np.ndarray): Cluster assignment for each sample.

        Returns:
            np.ndarray: Centroid matrix of shape (n_clusters, n_features).
        """
                                                                     
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features), dtype=float)
        for k in range(self.n_clusters):
            members = X[labels == k]
            if members.size == 0:
                continue
            counts = (members > 0).sum(axis=0)
            sums = members.sum(axis=0)                          
            with np.errstate(divide='ignore', invalid='ignore'):
                means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
            centroids[k] = means
        return centroids

    def _distances_to_centroids(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute assignment's Euclidean distances to each centroid with lambda masks.

        For each centroid, only coordinates where the centroid is non-zero are
        considered. For each sample, only its rated coordinates (non-zero) are
        included in the distance.

        Args:
            X (np.ndarray): Data matrix of shape (n_samples, n_features).
            centroids (np.ndarray): Centroid matrix of shape (k, n_features).

        Returns:
            np.ndarray: Distance matrix of shape (n_samples, k).
        """
        n_samples = X.shape[0]
        k = centroids.shape[0]
        dmat = np.empty((n_samples, k), dtype=float)
        large = 1e9  # sentinel for "very far" when there is no overlap
        for j in range(k):
            c = centroids[j]
            mc = c > 0
            if not np.any(mc):
                # No defined dimensions for this centroid
                dmat[:, j] = large
                continue
            X_sub = X[:, mc]
            mu_sub = X_sub > 0  # user lambda: consider only rated coords
            diff = X_sub - c[mc]
            sq = (diff * diff) * mu_sub
            sums = sq.sum(axis=1)
            overlaps = mu_sub.sum(axis=1)  # number of common rated items
            dist = np.sqrt(sums)
            dist = np.where(overlaps == 0, large, dist)  # far if no common ratings
            dmat[:, j] = dist
        return dmat

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run K-means with masked distances and return the best assignment.

        Args:
            X (np.ndarray): Data matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Cluster labels of shape (n_samples,).
        """
        rng = np.random.default_rng(self.random_state)
        X = X.astype(float, copy=False)
        n_samples = X.shape[0]
        best_inertia = np.inf  # keep the best run across n_init restarts
        best_labels = None
        best_centroids = None

        for init_idx in range(self.n_init):
            non_zero_rows = np.where((X > 0).sum(axis=1) > 0)[0]
            if len(non_zero_rows) >= self.n_clusters:
                seeds = rng.choice(non_zero_rows, size=self.n_clusters, replace=False)
            else:
                seeds = rng.choice(np.arange(n_samples), size=self.n_clusters, replace=False)
            centroids = X[seeds].copy()

            labels = None  # track to detect label convergence
            for it in range(self.max_iter):
                dmat = self._distances_to_centroids(X, centroids)
                new_labels = np.argmin(dmat, axis=1)                                         
                if labels is not None and np.array_equal(new_labels, labels):
                    labels = new_labels
                    break
                labels = new_labels
                new_centroids = self._compute_centroids(X, labels)
                shift = np.linalg.norm(new_centroids - centroids)  # centroid movement
                centroids = new_centroids
                if shift < self.tol:
                    break
                                                       
            final_dmat = self._distances_to_centroids(X, centroids)
            row_idx = np.arange(n_samples)
            inertia = np.sum(final_dmat[row_idx, labels] ** 2)  # masked SSE
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()

        self.labels_ = best_labels
        self.cluster_centers_ = best_centroids
        return self.labels_

def cluster_users_custom(R_matrix, n_clusters=5, metric="euclidean_custom"):
    """
    Cluster users with custom distance metrics for pattern recognition using K-means.

    Args:
        R_matrix (pd.DataFrame): User-item preference matrix
        n_clusters (int): Number of clusters
        metric (str): "euclidean_custom" or "cosine"

    Returns:
        tuple: (cluster_labels, clustering_model)
    """
    print(f"Clustering users with {metric} distance using K-means (k={n_clusters})...")
    X = R_matrix.values

    if metric == "euclidean_custom":
        print("Using CustomKMeansMasked (assignment's Euclidean with lambda masks)...")
        model = CustomKMeansMasked(n_clusters=n_clusters, n_init=5, max_iter=100, tol=1e-4, random_state=42)
        labels = model.fit_predict(X)

    elif metric == "cosine":
        print("Using spherical K-means (true cosine-based clustering)...")
        labels, centroids = spherical_kmeans(X, n_clusters=n_clusters, n_init=10, max_iter=100, tol=1e-4, random_state=42)
        class _SphericalModel:
            def __init__(self, centroids):
                self.cluster_centers_ = centroids
        model = _SphericalModel(centroids) 
    else:
        raise ValueError("Unsupported metric. Use 'euclidean_custom' or 'cosine'.")

    print(f"✓ K-means clustering completed with {n_clusters} clusters")
    return labels, model

def jaccard_distance(u_vec, v_vec):
    """
    Compute Jaccard distance between two users based on rated items only.

    Args:
        u_vec (np.array): Ratings vector of user u
        v_vec (np.array): Ratings vector of user v

    Returns:
        float: Jaccard distance
    """
    u_rated = u_vec > 0
    v_rated = v_vec > 0
    intersection = np.sum(u_rated & v_rated)
    union = np.sum(u_rated | v_rated)
    if union == 0:
        return 1.0
    return 1.0 - (intersection / union)

def _init_medoids_kpp(distance_matrix: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """
    Initialize medoids using a K-means++-style D^2 sampling on distances.

    Args:
        distance_matrix (np.ndarray): Precomputed pairwise distances, shape (n, n).
        n_clusters (int): Number of clusters (medoids) to select.
        rng (np.random.Generator): Random generator for reproducibility.

    Returns:
        np.ndarray: Array of selected medoid indices of shape (n_clusters,).
    """
    n = distance_matrix.shape[0]
    medoids = []                             
    first = rng.integers(0, n)  # pick initial medoid uniformly at random
    medoids.append(first)

    while len(medoids) < n_clusters:
                                             
        # Distance to closest existing medoid for each point
        dmin = np.min(distance_matrix[:, medoids], axis=1)
        dmin[medoids] = 0.0
        # Sample new medoid proportional to distance (D^2-like heuristic)
        probs = dmin / (dmin.sum() + 1e-12)
        next_idx = rng.choice(n, p=probs)
        if next_idx not in medoids:
            medoids.append(next_idx)
    return np.array(medoids, dtype=int)

def _assign_to_medoids(distance_matrix: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    """
    Assign each point to the nearest medoid under a precomputed metric.

    Args:
        distance_matrix (np.ndarray): Pairwise distances, shape (n, n).
        medoids (np.ndarray): Indices of current medoids, shape (k,).

    Returns:
        np.ndarray: Label array of shape (n,) with values in [0, k-1].
    """
    d_to_medoids = distance_matrix[:, medoids]
    return np.argmin(d_to_medoids, axis=1)

def _update_medoids(distance_matrix: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Update medoids by minimizing total in-cluster distance for each cluster.

    Args:
        distance_matrix (np.ndarray): Pairwise distances, shape (n, n).
        labels (np.ndarray): Current labels, shape (n,).
        n_clusters (int): Number of clusters.

    Returns:
        np.ndarray: New medoid indices of shape (n_clusters,). Empty clusters return -1.
    """
    new_medoids = np.empty(n_clusters, dtype=int)
    for k in range(n_clusters):
        members = np.where(labels == k)[0]
        if members.size == 0:
            new_medoids[k] = -1
            continue
        sub_d = distance_matrix[np.ix_(members, members)]
        costs = sub_d.sum(axis=1)
        best_local = members[np.argmin(costs)]
        new_medoids[k] = best_local
    return new_medoids

def kmedoids(distance_matrix: np.ndarray, n_clusters: int, max_iter: int = 100, random_state: int = 42):
    """
    Partition data into k clusters around medoids under a precomputed metric.

    A lightweight implementation of K-Medoids that supports empty cluster
    handling via reseeding to farthest non-medoid points.

    Args:
        distance_matrix (np.ndarray): Pairwise distances, shape (n, n).
        n_clusters (int): Desired number of clusters.
        max_iter (int, optional): Maximum iterations. Defaults to 100.
        random_state (int, optional): RNG seed. Defaults to 42.

    Returns:
        tuple[np.ndarray, Any]: (labels, model) where labels has shape (n,) and
        model exposes `medoid_indices_` and `metric` attributes.
    """
    rng = np.random.default_rng(random_state)
    n = distance_matrix.shape[0]
    medoids = _init_medoids_kpp(distance_matrix, n_clusters, rng)

    for _ in range(max_iter):
        labels = _assign_to_medoids(distance_matrix, medoids)
        new_medoids = _update_medoids(distance_matrix, labels, n_clusters)

        for i in range(n_clusters):
            if new_medoids[i] == -1:
                # Re-seed empty cluster to farthest point from current medoids
                non_medoids = np.setdiff1d(np.arange(n), medoids, assume_unique=False)
                dmin = np.min(distance_matrix[np.ix_(non_medoids, medoids)], axis=1)
                reseed = non_medoids[np.argmax(dmin)]
                new_medoids[i] = reseed

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    labels = _assign_to_medoids(distance_matrix, medoids)
    class _KMedoidsModel:
        def __init__(self, medoids):
            self.medoid_indices_ = medoids
            self.metric = "precomputed"
    return labels, _KMedoidsModel(medoids)

def cluster_users_jaccard(R_matrix, n_clusters=5):
    """
    Cluster users using Jaccard distance on rated items with K-Medoids.

    Args:
        R_matrix (pd.DataFrame): User-item preference matrix
        n_clusters (int): Number of clusters

    Returns:
        tuple: (labels, model)
    """
    print(f"Clustering users with Jaccard distance using K-Medoids (k={n_clusters})...")
    X_bin = (R_matrix.values > 0)  # presence/absence of ratings
    dist_matrix = pairwise_distances(X_bin, metric='jaccard')  # Jaccard distance on binary masks
    labels, model = kmedoids(dist_matrix, n_clusters=n_clusters, max_iter=100, random_state=42)
    print(f"✓ K-Medoids clustering completed with {n_clusters} clusters (Jaccard)")
    return labels, model

                                                                   
def find_k_nearest_neighbors(R_matrix: pd.DataFrame, user_id, cluster_users, k: int = 3):
    """
    Find the k nearest neighbors of a user within a cluster using Jaccard distance.

    Args:
        R_matrix (pd.DataFrame): User–item matrix indexed by user_id.
        user_id: Target user identifier present in `R_matrix.index`.
        cluster_users (Iterable): User ids belonging to the same cluster.
        k (int, optional): Number of neighbors to return. Defaults to 3.

    Returns:
        list: List of neighbor user ids ordered by increasing distance.
    """
    target_vec = R_matrix.loc[user_id].values
    distances = []
    for other in cluster_users:
        if other == user_id:
            continue
        other_vec = R_matrix.loc[other].values
        dist = jaccard_distance(target_vec, other_vec)
        distances.append((other, dist))
    neighbors = sorted(distances, key=lambda x: x[1])[:k]
    return [n[0] for n in neighbors]

def build_cluster_nn_data_latent(R_matrix: pd.DataFrame, labels: np.ndarray, k: int = 3,
                                 svd_components: int = 64, random_state: int = 42):
    """
    Build per-cluster training data in a reduced latent space using TruncatedSVD.
    Center ratings by user mean (on rated entries) before SVD.
    X_row = concat([latent(user), latent(neighbor1), ..., latent(neighbork)])
    y_row = latent(user)
    """
    print(f"Building latent features with TruncatedSVD (components={svd_components}) and user-mean centering...")
    R_values = R_matrix.values.astype(float)
    rated_mask = R_values > 0  # treat 0 as missing ratings
    with np.errstate(invalid='ignore'):
        user_means = np.divide(R_values.sum(axis=1), rated_mask.sum(axis=1), out=np.zeros(R_values.shape[0]), where=rated_mask.sum(axis=1) > 0)
    R_centered = R_values.copy()
    for i in range(R_centered.shape[0]):
        if rated_mask[i].any():
            # Center only on rated entries; keep missing entries at 0
            R_centered[i, rated_mask[i]] = R_centered[i, rated_mask[i]] - user_means[i]
    svd = TruncatedSVD(n_components=svd_components, random_state=random_state)
    user_latent = svd.fit_transform(R_centered)                       

    users = R_matrix.index.tolist()
    clusters = {}
    for u, lbl in zip(users, labels):
        clusters.setdefault(lbl, []).append(u)

    cluster_data = {}
    for cluster_id, cluster_users in clusters.items():
        X_list, y_list, uid_list = [], [], []
        user_idx = {u: i for i, u in enumerate(users)}
        for u in cluster_users:
            neighs = find_k_nearest_neighbors(R_matrix, u, cluster_users, k=k)
            idx_u = user_idx[u]
            parts = [user_latent[idx_u]]  # target user embedding
            for v in neighs:
                parts.append(user_latent[user_idx[v]])  # neighbor embeddings
            X_list.append(np.concatenate(parts))
            y_list.append(user_latent[idx_u])  # predict the target user's latent vector
            uid_list.append(u)
        cluster_data[cluster_id] = {
            'X': np.array(X_list, dtype=float),
            'y': np.array(y_list, dtype=float),
            'users': uid_list
        }
    print("✓ Built per-cluster latent training datasets")
    return cluster_data, svd

def train_cluster_mlps(cluster_data: dict, hidden_layer_sizes=(256, 128), max_iter: int = 1500,
                       random_state: int = 42, learning_rate_init: float = 5e-4, alpha: float = 1e-4):
    """
    Train one MLP regressor per cluster to predict latent user vectors.

    Args:
        cluster_data (dict): Mapping cluster_id -> {'X','y','users'} arrays.
        hidden_layer_sizes (tuple): MLP hidden sizes. Defaults to (256, 128).
        max_iter (int): Max training iterations. Defaults to 1500.
        random_state (int): RNG seed. Defaults to 42.
        learning_rate_init (float): Initial learning rate. Defaults to 5e-4.
        alpha (float): L2 regularization. Defaults to 1e-4.

    Returns:
        dict: Mapping cluster_id -> fitted MLPRegressor.
    """
    print("Training MLP models per cluster (predicting latent user vectors)...")
    models = {}
    for cid, data in cluster_data.items():
        X = data['X']
        y = data['y']
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            early_stopping=True,
            n_iter_no_change=20,
            tol=1e-4,
            validation_fraction=0.1,
        )
        model.fit(X, y)
        models[cid] = model
        print(f"  ✓ Trained cluster {cid}: X={X.shape}, y={y.shape}")
    return models

def evaluate_models(models: dict, cluster_data: dict, svd: TruncatedSVD, R_matrix: pd.DataFrame):
    """
    Reconstruct ratings from predicted latent vectors and report RMSE over rated entries.
    """
    print("Evaluating per-cluster models (RMSE on rated entries)...")
    components = svd.components_                       
    users_all = R_matrix.index.tolist()
    user_index = {u: i for i, u in enumerate(users_all)}

    rmses = []
    for cid, model in models.items():
        data = cluster_data[cid]
        X = data['X']
        uids = data['users']
        y_lat_pred = model.predict(X)                 
        R_pred = y_lat_pred @ components
        err_sum = 0.0
        count = 0
        for row_idx, u in enumerate(uids):
            orig = R_matrix.loc[u].values
            mask = orig > 0
            if not np.any(mask):
                continue
            diff = R_pred[row_idx, mask] - orig[mask]
            err_sum += np.sum(diff * diff)
            count += mask.sum()
        if count > 0:
            rmse = np.sqrt(err_sum / count)
            rmses.append(rmse)
            print(f"  Cluster {cid}: RMSE={rmse:.3f} over {count} rated entries")
    if rmses:
        print(f"✓ Mean RMSE across clusters: {np.mean(rmses):.3f}")

def split_cluster_data(cluster_data: dict, test_size: float = 0.2, random_state: int = 42):
    """
    Split each cluster's dataset into train/test keeping user id order alignment.
    Returns a dict with train/test splits and user ids for evaluation.
    """
    split = {}
    for cid, data in cluster_data.items():
        X = data['X']
        y = data['y']
        users = data['users']
        X_train, X_test, y_train, y_test, users_train, users_test = train_test_split(
            X, y, users, test_size=test_size, random_state=random_state
        )
        split[cid] = {
            'train': {'X': X_train, 'y': y_train, 'users': users_train},
            'test': {'X': X_test, 'y': y_test, 'users': users_test},
        }
    return split

def evaluate_models_on_test(models: dict, split_data: dict, svd: TruncatedSVD, R_matrix: pd.DataFrame):
    """
    Evaluate per-cluster models on held-out users and report RMSE on rated items.

    Predictions are reconstructed from latent space and re-centered by user means
    before clipping to the rating bounds [1, 10].

    Args:
        models (dict): Mapping cluster_id -> trained model.
        split_data (dict): Output of `split_cluster_data` with test sets.
        svd (TruncatedSVD): Fitted SVD model providing components_.
        R_matrix (pd.DataFrame): Original user–item matrix for evaluation.
    """
    print("Evaluating per-cluster models on held-out test users (RMSE on rated entries)...")
    components = svd.components_
    rmses = []
    R_values = R_matrix.values.astype(float)
    rated_mask = R_values > 0  # identify rated entries per user
    with np.errstate(invalid='ignore'):
        user_means = np.divide(R_values.sum(axis=1), rated_mask.sum(axis=1), out=np.zeros(R_values.shape[0]), where=rated_mask.sum(axis=1) > 0)
    users_all = R_matrix.index.tolist()
    user_to_idx = {u: i for i, u in enumerate(users_all)}
    for cid, model in models.items():
        test = split_data[cid]['test']
        X = test['X']
        uids = test['users']
        if len(uids) == 0:
            continue
        y_lat_pred = model.predict(X)
        R_pred_centered = y_lat_pred @ components  # map latent to item space
        means_vec = np.array([user_means[user_to_idx[u]] for u in uids], dtype=float)
        R_pred = R_pred_centered + means_vec[:, None]  # add back user mean
        R_pred = np.clip(R_pred, 1.0, 10.0)  # clamp to rating scale
        err_sum = 0.0
        count = 0
        for row_idx, u in enumerate(uids):
            orig = R_matrix.loc[u].values
            mask = orig > 0
            if not np.any(mask):
                continue
            diff = R_pred[row_idx, mask] - orig[mask]
            err_sum += np.sum(diff * diff)
            count += mask.sum()
        if count > 0:
            rmse = np.sqrt(err_sum / count)
            rmses.append(rmse)
            print(f"  Cluster {cid}: Test RMSE={rmse:.3f} over {count} rated entries")
    if rmses:
        print(f"✓ Mean Test RMSE across clusters: {np.mean(rmses):.3f}")

def compute_mae_train_test(models: dict, split_data: dict, svd: TruncatedSVD, R_matrix: pd.DataFrame):
    """
    Compute MAE on rated entries for both train and test splits per cluster.
    Returns dict: {cluster_id: (train_mae, test_mae)} and prints a table.
    """
    print("\nEvaluating MAE on train and test (rated entries only)...")
    components = svd.components_
    results = {}

    def _mae_for(users, R_pred):
        abs_sum = 0.0
        count = 0
        for row_idx, u in enumerate(users):
            orig = R_matrix.loc[u].values
            mask = orig > 0
            if not np.any(mask):
                continue
            # Only compare on rated entries
            abs_sum += np.sum(np.abs(R_pred[row_idx, mask] - orig[mask]))
            count += int(mask.sum())
        return (abs_sum / count) if count > 0 else np.nan

    R_values = R_matrix.values.astype(float)
    rated_mask = R_values > 0  # identify rated entries per user
    with np.errstate(invalid='ignore'):
        user_means = np.divide(R_values.sum(axis=1), rated_mask.sum(axis=1), out=np.zeros(R_values.shape[0]), where=rated_mask.sum(axis=1) > 0)
    users_all = R_matrix.index.tolist()
    user_to_idx = {u: i for i, u in enumerate(users_all)}

    for cid, model in models.items():
        tr = split_data[cid]['train']
        te = split_data[cid]['test']
        y_lat_pred_tr = model.predict(tr['X']) if len(tr['users']) > 0 else np.empty((0, components.shape[0]))
        R_pred_tr_c = y_lat_pred_tr @ components if y_lat_pred_tr.size else y_lat_pred_tr
        if y_lat_pred_tr.size:
            means_vec_tr = np.array([user_means[user_to_idx[u]] for u in tr['users']], dtype=float)
            R_pred_tr = np.clip(R_pred_tr_c + means_vec_tr[:, None], 1.0, 10.0)  # add mean, clamp
        else:
            R_pred_tr = R_pred_tr_c
        train_mae = _mae_for(tr['users'], R_pred_tr)
        y_lat_pred_te = model.predict(te['X']) if len(te['users']) > 0 else np.empty((0, components.shape[0]))
        R_pred_te_c = y_lat_pred_te @ components if y_lat_pred_te.size else y_lat_pred_te
        if y_lat_pred_te.size:
            means_vec_te = np.array([user_means[user_to_idx[u]] for u in te['users']], dtype=float)
            R_pred_te = np.clip(R_pred_te_c + means_vec_te[:, None], 1.0, 10.0)  # add mean, clamp
        else:
            R_pred_te = R_pred_te_c
        test_mae = _mae_for(te['users'], R_pred_te)
        results[cid] = (train_mae, test_mae)

    print("\nMAE Results per Cluster:")
    print("Cluster | Train MAE | Test MAE")
    for cid in sorted(results.keys()):
        tr_mae, te_mae = results[cid]
        tr_txt = f"{tr_mae:.4f}" if not np.isnan(tr_mae) else "NA"
        te_txt = f"{te_mae:.4f}" if not np.isnan(te_mae) else "NA"
        print(f"{cid:7d} | {tr_txt:9s} | {te_txt}")
    tr_vals = [v[0] for v in results.values() if not np.isnan(v[0])]
    te_vals = [v[1] for v in results.values() if not np.isnan(v[1])]
    if tr_vals:
        print(f"\n✓ Mean Train MAE: {np.mean(tr_vals):.4f}")
    if te_vals:
        print(f"✓ Mean Test MAE: {np.mean(te_vals):.4f}")
    return results

def plot_clusters(R_matrix, labels, title="User Clusters"):
    """
    Visualize user clusters using 3D dimensionality reduction.

    Args:
        R_matrix (pd.DataFrame): User-item preference matrix
        labels (np.array): Cluster labels for each user
        title (str): Plot title
    """
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    pca = PCA(n_components=3, random_state=42)
    X_3d = pca.fit_transform(R_matrix.values)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for i, (label, color) in enumerate(zip(unique_labels, colors)):
        mask = labels == label
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                  c=[color], label=f'Cluster {label}',
                  alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

    ax.set_title(f'{title} (3D PCA Projection)', fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)

    variance_explained = pca.explained_variance_ratio_
    ax.text2D(0.05, 0.95,
             '.1f',
             transform=ax.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    max_range = np.array([X_3d[:, 0].max()-X_3d[:, 0].min(),
                         X_3d[:, 1].max()-X_3d[:, 1].min(),
                         X_3d[:, 2].max()-X_3d[:, 2].min()]).max() / 2.0

    mid_x = (X_3d[:, 0].max()+X_3d[:, 0].min()) * 0.5
    mid_y = (X_3d[:, 1].max()+X_3d[:, 1].min()) * 0.5
    mid_z = (X_3d[:, 2].max()+X_3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '')}_3d.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 3D Cluster visualization saved to: {os.path.abspath(plot_path)}")

    plt.close()

def run_clustering_experiments(R_matrix, cluster_values=[2, 3, 5, 10]):
    """
    Run K-means clustering experiments for different cluster numbers (L) and metrics.
    Implements Query 1(c): Test different values of parameter L.

    Args:
        R_matrix (pd.DataFrame): User-item preference matrix
        cluster_values (list): List of cluster numbers L to test
    """
    print("\n=== QUERY 1(c): K-MEANS CLUSTERING EXPERIMENTS FOR DIFFERENT L VALUES ===")
    print(f"Testing L (number of clusters) = {cluster_values}")

    all_results = {}

    for L in cluster_values:
        print(f"\n--- K-means clustering with L={L} clusters ---")
        all_results[L] = {}

        print(f"Running K-means with Euclidean distance (L={L})...")
        labels_euc, _ = cluster_users_custom(R_matrix, n_clusters=L, metric="euclidean_custom")
        cluster_counts_euc = pd.Series(labels_euc).value_counts().sort_index()

        print(f"Cluster distribution (Euclidean, L={L}):")
        for cluster_id, count in cluster_counts_euc.items():
            print(f"  Cluster {cluster_id}: {count} users ({count/len(labels_euc)*100:.1f}%)")

        # Save a 3D PCA plot of clusters for visual inspection
        plot_clusters(R_matrix, labels_euc, title=f"K-means Clusters (Euclidean, L={L})")
        all_results[L]['euclidean'] = {'labels': labels_euc, 'counts': cluster_counts_euc}

        print(f"Running K-means with Cosine distance (L={L})...")
        labels_cos, _ = cluster_users_custom(R_matrix, n_clusters=L, metric="cosine")
        cluster_counts_cos = pd.Series(labels_cos).value_counts().sort_index()

        print(f"Cluster distribution (Cosine, L={L}):")
        for cluster_id, count in cluster_counts_cos.items():
            print(f"  Cluster {cluster_id}: {count} users ({count/len(labels_cos)*100:.1f}%)")

        # Save a 3D PCA plot of clusters for visual inspection
        plot_clusters(R_matrix, labels_cos, title=f"K-means Clusters (Cosine, L={L})")
        all_results[L]['cosine'] = {'labels': labels_cos, 'counts': cluster_counts_cos}

    print("\n✓ Query 1(c) K-means experiments completed!")
    print(f"✓ Generated {len(cluster_values) * 2} cluster visualizations")
    print("✓ All plots saved to output/ folder")

    return all_results

def main():
    """
    Main function to implement Query 2 preprocessing: filter users and movies.
    """
    dataset_path = "data/Dataset.npy"

    dataset = load_npy_dataset(dataset_path)

    if dataset is not None:
        # Convert raw lines to a typed DataFrame with parsed dates
        df = parse_ratings_to_dataframe(dataset)

        if df is not None:
            U, I, num_users, num_movies = find_unique_users_and_movies(df)

            print("=== QUERY 1: ORIGINAL DATASET ANALYSIS ===")
            print(f"Number of unique users (U): {num_users}")
            print(f"Number of unique movies (I): {num_movies}")
            print(f"Total ratings: {len(df)}")

            # Choose a narrower band of active users for stable clustering
            R_MIN = 90                                              
            R_MAX = 100                                         

            print(f"\n=== QUERY 2: PREPROCESSING (FILTERING) ===")
            print(f"R_min = {R_MIN}, R_max = {R_MAX}")

            filtered_df, U_hat, I_hat, num_filtered_users, num_filtered_movies = filter_users_and_movies(df, R_MIN, R_MAX)

            print(f"\n=== FILTERED DATASET (Û, Î) ===")
            print(f"Number of filtered users (Û): {num_filtered_users}")
            print(f"Number of filtered movies (Î): {num_filtered_movies}")
            print(f"Total ratings after filtering: {len(filtered_df)}")

            users_removed = num_users - num_filtered_users
            movies_removed = num_movies - num_filtered_movies
            ratings_removed = len(df) - len(filtered_df)

            print(f"\n=== FILTERING STATISTICS ===")
            print(f"Users removed: {users_removed} ({users_removed/num_users*100:.1f}%)")
            print(f"Movies removed: {movies_removed} ({movies_removed/num_movies*100:.1f}%)")
            print(f"Ratings removed: {ratings_removed} ({ratings_removed/len(df)*100:.1f}%)")

            print(f"\n=== QUERY 3: HISTOGRAMS (RATING PATTERNS) ===")
            print("Creating histograms for rating patterns...")

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            histogram_path = os.path.join(output_dir, "user_rating_histograms.png")
            plot_histograms(filtered_df, save_path=histogram_path)

            print(f"✓ Histograms created and saved at: {os.path.abspath(histogram_path)}")

            print(f"\n=== HISTOGRAM STATISTICS ===")

            user_rating_counts = filtered_df.groupby('user_id').size()
            print(f"Rating Counts per User:")
            print(f"  Mean: {user_rating_counts.mean():.1f}")
            print(f"  Median: {user_rating_counts.median():.1f}")
            print(f"  Min: {user_rating_counts.min()}")
            print(f"  Max: {user_rating_counts.max()}")

            user_time_stats = filtered_df.groupby('user_id')['date'].agg(['min', 'max'])
            user_time_spans = (user_time_stats['max'] - user_time_stats['min']).dt.days
            user_time_spans = user_time_spans[user_time_spans > 0]

            if len(user_time_spans) > 0:
                print(f"\nTime Span per User (days):")
                print(f"  Mean: {user_time_spans.mean():.1f}")
                print(f"  Median: {user_time_spans.median():.1f}")
                print(f"  Min: {user_time_spans.min()}")
                print(f"  Max: {user_time_spans.max()}")

                print(f"  Mean in months: {user_time_spans.mean() / 30.44:.1f}")
                print(f"  Mean in years: {user_time_spans.mean() / 365.25:.2f}")

            print(f"\n=== QUERY 4: PREFERENCE MATRIX CONSTRUCTION ===")
            R_matrix = build_preference_matrix(filtered_df, U_hat, I_hat)

            print(f"\n=== MATRIX STATISTICS ===")
            print(f"Non-zero entries: {(R_matrix != 0).sum().sum()}")
            print(f"Density: {(R_matrix != 0).sum().sum() / R_matrix.size * 100:.2f}%")
            print(f"Memory usage: {R_matrix.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

            print(f"\n=== QUERY 1: PATTERN RECOGNITION - CLUSTERING ANALYSIS ===")

            sample_size = min(800, len(R_matrix))  # keep runtime manageable
            if len(R_matrix) > sample_size:
                print(f"Sampling {sample_size} users out of {len(R_matrix)} for faster clustering experiments...")
                sample_indices = np.random.choice(len(R_matrix), size=sample_size, replace=False)
                R_sample = R_matrix.iloc[sample_indices]
                print(f"Sample matrix shape: {R_sample.shape}")
            else:
                R_sample = R_matrix
                sample_size = len(R_matrix)

            cluster_values = [2, 3, 5, 10]                                      
            experiment_results = run_clustering_experiments(R_sample, cluster_values)

            print(f"\n✓ Pattern Recognition Analysis Complete!")
            print(f"✓ Query 1(c) K-means experiments completed with L = {cluster_values}")
            print(f"✓ Generated {len(cluster_values) * 2} cluster visualizations")
            print(f"✓ All results saved to output/ folder")

            print(f"\n=== QUERY 2: CLUSTERING WITH JACCARD DISTANCE (K-MEDOIDS) ===")
            L_jaccard = 5  # number of clusters for Jaccard + K-medoids
            labels_jac, model_jac = cluster_users_jaccard(R_sample, n_clusters=L_jaccard)
            cluster_counts_jac = pd.Series(labels_jac).value_counts().sort_index()
            print("Cluster distribution (Jaccard):")
            for cluster_id, count in cluster_counts_jac.items():
                print(f"  Cluster {cluster_id}: {count} users ({count/len(labels_jac)*100:.1f}%)")

            plot_clusters(R_sample, labels_jac, title=f"K-medoids Clusters (Jaccard, L={L_jaccard})")

            print(f"\n=== QUERY 2(b): NEIGHBORS + PER-CLUSTER MLP TRAINING ===")
            cluster_datasets, svd_model = build_cluster_nn_data_latent(R_sample, labels_jac, k=5, svd_components=128)
            split_data = split_cluster_data(cluster_datasets, test_size=0.2)
            train_only = {cid: split_data[cid]['train'] for cid in split_data}
            # Train MLPs to predict latent user vectors from user+neighbor latents
            mlp_models = train_cluster_mlps(train_only, hidden_layer_sizes=(256, 128), max_iter=1500, learning_rate_init=5e-4, alpha=1e-4)
            evaluate_models_on_test(mlp_models, split_data, svd_model, R_sample)
            compute_mae_train_test(mlp_models, split_data, svd_model, R_sample)

            return filtered_df, U_hat, I_hat, df, U, I, R_matrix, experiment_results

if __name__ == "__main__":
    main()
