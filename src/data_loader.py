import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def clean_and_preprocess_dataframe(df_path, sample_limit=None):
    """
    Loads raw CIC-IDS2017 flow CSV, handles Infs, NaNs, and strips whitespace.
    Returns:
        X (np.ndarray): Numerical feature matrix
        y (np.ndarray): Binary labels (0 = Benign/Normal, 1 = Attack)
        feature_names (list): Names of used features
    """
    print(f"[*] Reading dataset from {df_path}...")
    df = pd.read_csv(df_path)
    df.columns = [c.strip() for c in df.columns]

    # Identify label column
    label_col = [c for c in df.columns if "label" in c.lower()][0]

    # Binary label encoding: BENIGN -> 0, Attack -> 1
    y_raw = df[label_col].astype(str).str.strip().str.upper()
    y = (y_raw != "BENIGN").astype(int).values

    # Feature matrix
    X_df = df.drop(columns=[label_col])

    # Replace inf and -inf with NaN
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    # Fill NaNs with column median (calculated per column)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    # Remove zero-variance (constant) columns
    non_constant_cols = [c for c in X_df.columns if X_df[c].nunique() > 1]
    X_df = X_df[non_constant_cols]

    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)

    if sample_limit is not None and sample_limit < len(X):
        # Subsample maintaining class ratio
        np.random.seed(42)
        idx = np.random.choice(len(X), size=sample_limit, replace=False)
        X = X[idx]
        y = y[idx]

    print(f"[+] Loaded {len(X)} records across {len(feature_names)} features.")
    print(f"    - Benign flows: {int((y == 0).sum())} ({((y == 0).sum() / len(y) * 100):.2f}%)")
    print(f"    - Attack flows: {int((y == 1).sum())} ({((y == 1).sum() / len(y) * 100):.2f}%)")

    return X, y, feature_names


def prepare_federated_splits(X, y, num_clients=10, non_iid=False, alpha=0.5, random_seed=42):
    """
    Splits dataset following Fed-ANIDS protocol:
    1. Only BENIGN flows are used for training and validation.
    2. Benign flows are split: 70% Train, 10% Validation, 20% Test.
    3. Test set contains the remaining 20% Benign flows + all Attack flows.
    4. Train Benign flows are partitioned across `num_clients` (IID or Non-IID Dirichlet).
    5. Normalization (MinMaxScaler) is fitted strictly on training benign data to avoid leakage.
    """
    np.random.seed(random_seed)

    # Separate benign and attack
    benign_mask = (y == 0)
    X_benign = X[benign_mask]
    y_benign = y[benign_mask]

    X_attack = X[~benign_mask]
    y_attack = y[~benign_mask]

    # Split benign flows: 70% train, 10% val, 20% test
    X_b_train, X_b_rem, y_b_train, y_b_rem = train_test_split(
        X_benign, y_benign, test_size=0.30, random_state=random_seed
    )
    X_b_val, X_b_test, y_b_val, y_b_test = train_test_split(
        X_b_rem, y_b_rem, test_size=(2.0 / 3.0), random_state=random_seed
    )

    # Fit MinMaxScaler ONLY on benign training data
    scaler = MinMaxScaler()
    X_b_train_scaled = scaler.fit_transform(X_b_train).astype(np.float32)
    X_b_val_scaled = scaler.transform(X_b_val).astype(np.float32)
    X_b_test_scaled = scaler.transform(X_b_test).astype(np.float32)
    X_attack_scaled = scaler.transform(X_attack).astype(np.float32)

    # Construct complete Test Set: Benign test + Attack
    X_test_scaled = np.vstack([X_b_test_scaled, X_attack_scaled])
    y_test = np.concatenate([y_b_test, y_attack])

    # Shuffle test set
    perm = np.random.permutation(len(y_test))
    X_test_scaled = X_test_scaled[perm]
    y_test = y_test[perm]

    # Partition training benign data among clients
    client_datasets = []
    n_train = len(X_b_train_scaled)

    if not non_iid:
        indices = np.random.permutation(n_train)
        client_chunks = np.array_split(indices, num_clients)
        for chunk in client_chunks:
            client_datasets.append(X_b_train_scaled[chunk])
    else:
        # Balanced Dirichlet to ensure realistic heterogeneity while guaranteeing minimum size
        min_samples = 100
        available_samples = n_train - (min_samples * num_clients)
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * available_samples).astype(int) + min_samples
        counts[-1] = n_train - counts[:-1].sum()

        indices = np.random.permutation(n_train)
        start = 0
        for count in counts:
            chunk = indices[start : start + count]
            client_datasets.append(X_b_train_scaled[chunk])
            start += count

    print(f"\n[+] Federated Split Summary ({num_clients} Clients):")
    print(f"    - Benign Training flows (total): {n_train}")
    print(f"    - Benign Validation flows (for threshold tuning): {len(X_b_val_scaled)}")
    print(f"    - Evaluation Test flows: {len(X_test_scaled)} (Benign: {len(y_b_test)}, Attack: {len(y_attack)})")
    for i, c_data in enumerate(client_datasets):
        print(f"      Client Node {i+1:02d}: {len(c_data):5d} benign samples")

    return {
        "client_datasets": client_datasets,
        "val_data": X_b_val_scaled,
        "test_X": X_test_scaled,
        "test_y": y_test,
        "scaler": scaler,
        "input_dim": X.shape[1]
    }


def get_dataloader(data_array, batch_size=64, shuffle=True):
    """Creates a PyTorch DataLoader from a numpy array with safe batching."""
    tensor = torch.tensor(data_array, dtype=torch.float32)
    dataset = TensorDataset(tensor, tensor)
    # Drop last batch if it contains only 1 sample to avoid normalization issues
    drop_last = (len(data_array) > batch_size) and (len(data_array) % batch_size == 1)
    actual_batch_size = min(batch_size, len(data_array))
    return DataLoader(dataset, batch_size=actual_batch_size, shuffle=shuffle, drop_last=drop_last)
