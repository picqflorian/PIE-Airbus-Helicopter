# ============================================================
# 0. Imports & Options
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import groupby
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


# ============================================================
# 1. Flight → Temporal Symbolic Sequences
# ============================================================

def process_flight_sequences(df, md=1.0):
    """
    Converts flight data into temporal symbolic sequences.
    Duration is encoded via repetition.
    """
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # Sort chronologically
    df = df.sort_values(by=['F_SESSION', 'F_START_FRAME'])

    # --- Phase ID construction ---
    # Ensure indices are integers, then strings, then zero-padded
    df['FIRST_WORD_INDEX']  = df['FIRST_WORD_INDEX'].astype(int).astype(str).str.zfill(3)
    df['SECOND_WORD_INDEX'] = df['SECOND_WORD_INDEX'].astype(int).astype(str).str.zfill(3)
    df['THIRD_WORD_INDEX']  = df['THIRD_WORD_INDEX'].astype(int).astype(str).str.zfill(4)

    df['Phase_ID'] = (
        df['FIRST_WORD_INDEX'] +
        df['SECOND_WORD_INDEX'] +
        df['THIRD_WORD_INDEX']
    )

    # --- Duration handling ---
    # Convert F_DURATION to timedelta if it's a string, or handle accordingly
    # Assuming F_DURATION is in a standard time format or convertible string
    df['F_DURATION'] = pd.to_timedelta(df['F_DURATION'])
    df['duration_sec'] = df['F_DURATION'].dt.total_seconds()

    if md is not None:
        df['reps'] = np.floor(df['duration_sec'] / md).astype(int)
    else:
        df['reps'] = 1
    
    # Ensure at least 1 repetition so short phases aren't lost
    df.loc[df['reps'] < 1, 'reps'] = 1
    df = df[df['reps'] > 0]

    # --- Temporal expansion (Sequence Generation) ---
    df_expanded = df.loc[df.index.repeat(df['reps'])].copy()
    df_expanded['step'] = df_expanded.groupby('F_SESSION').cumcount()

    # --- Final sequence table (Pivot) ---
    df_sequence = df_expanded.pivot(
        index='F_SESSION',
        columns='step',
        values='Phase_ID'
    )

    return df_sequence


# ============================================================
# 2. Logical Waypoints with Duration (Compression)
# ============================================================

def compress_sequence_with_duration(sequence):
    """
    Compresses a symbolic sequence into (Phase_ID, duration) tuples.
    Example: [A, A, A, B, B] -> [(A, 3), (B, 2)]
    """
    return [(phase, len(list(group)))
            for phase, group in groupby(sequence)]


def build_logical_sequences(df_sequence):
    """
    Applies logical compression to all flights in the dataframe.
    """
    return df_sequence.apply(
        lambda row: compress_sequence_with_duration(row.dropna().tolist()),
        axis=1
    )


# ============================================================
# 3. Weighted LCS (Duration-Aware)
# ============================================================

def weighted_lcs(seq1, seq2):
    """
    Computes duration-weighted LCS between two logical sequences.
    """
    m, n = len(seq1), len(seq2)
    # DP Matrix initialization
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            p1, d1 = seq1[i]
            p2, d2 = seq2[j]

            if p1 == p2:
                # If phase matches, add the minimum shared duration
                dp[i + 1][j + 1] = dp[i][j] + min(d1, d2)
            else:
                # If not, carry over the max value from neighbors
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[m][n]


def weighted_lcs_similarity(seq1, seq2):
    """
    Normalized similarity score in [0, 1].
    """
    # Normalization: divide by the minimum total duration of the two flights
    len1 = sum(d for _, d in seq1)
    len2 = sum(d for _, d in seq2)
    
    total_duration = min(len1, len2)

    if total_duration == 0:
        return 0.0

    return weighted_lcs(seq1, seq2) / total_duration


# ============================================================
# 4. Similarity Matrix Calculation
# ============================================================

def build_similarity_matrix(logical_sequences):
    """
    Computes pairwise weighted LCS similarity matrix.
    NOTE: This is O(N^2) complexity. 
    """
    flights = logical_sequences.index.tolist()
    n = len(flights)

    sim_matrix = np.zeros((n, n))
    
    print(f"Calculating similarity matrix for {n} flights...")

    for i in range(n):
        if i % 10 == 0: 
            print(f"Processing row {i}/{n}")
            
        # Matrix is symmetric, compute only upper triangle
        for j in range(i, n): 
            print(f"i = {i}, j = {j}") 
            score = weighted_lcs_similarity(
                logical_sequences.iloc[i],
                logical_sequences.iloc[j]
            )
            sim_matrix[i, j] = score
            sim_matrix[j, i] = score

    return pd.DataFrame(sim_matrix, index=flights, columns=flights)


# ============================================================
# 5. Visualization & Cluster Analysis
# ============================================================

def visualize_dendrogram(linkage_matrix, labels):
    """
    Plots the Hierarchical Clustering Dendrogram.
    """
    plt.figure(figsize=(12, 5))
    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.ylabel("Distance (1 - Similarity)")
    plt.xlabel("Flights (F_SESSION)")
    plt.tight_layout()
    plt.show()


def analyze_clusters_at_thresholds(linkage_matrix, sim_matrix_index, df_context, thresholds):
    """
    Cuts the dendrogram at specified thresholds and analyzes cluster composition
    against metadata (Aircraft, Operator, Mission).
    """
    
    # Ensure the indices match (sim_matrix_index usually refers to F_SESSION)
    # Join logic: cluster labels -> F_SESSION -> Context Metadata
    
    for t in thresholds:
        print("\n" + "="*80)
        print(f"CUTTING DENDROGRAM AT THRESHOLD (DISTANCE) = {t}")
        print("="*80)
        
        # 'fcluster' assigns cluster labels based on the distance threshold
        cluster_labels = fcluster(linkage_matrix, t=t, criterion='distance')
        
        # Create a results DataFrame
        res = pd.DataFrame(index=sim_matrix_index)
        res['Cluster'] = cluster_labels
        
        # Merge with context metadata
        res = res.join(df_context, how='left')
        
        num_clusters = res['Cluster'].nunique()
        print(f"-> Number of clusters generated: {num_clusters}")
        
        # --- Analysis ---
        
        # 1. Cluster Sizes
        print("\n--- Cluster Sizes ---")
        print(res['Cluster'].value_counts().sort_index())
        
        # 2. Relationship with Aircraft
        if 'k_aircraft' in res.columns:
            print("\n--- Aircraft Distribution per Cluster ---")
            ct = pd.crosstab(res['Cluster'], res['k_aircraft'])
            print(ct)
            
        # 3. Relationship with Mission
        if 'k_mission' in res.columns:
            print("\n--- Mission Distribution per Cluster ---")
            ct_mission = pd.crosstab(res['Cluster'], res['k_mission'])
            print(ct_mission)

        # 4. Relationship with Operator
        if 'k_operator' in res.columns:
            print("\n--- Operator Distribution per Cluster ---")
            ct_op = pd.crosstab(res['Cluster'], res['k_operator'])
            print(ct_op)
            

# ============================================================
# 6. Main Execution Pipeline
# ============================================================

if __name__ == "__main__":

    # --- 1. Load Data ---
    filename = "PIE_data_with_context.csv"
    
    # WARNING: 'nrows=100' is set for quick testing/debugging. 
    # Weighted LCS is O(N^2). Removing this limit for 30k flights 
    # will take extremely long without optimization/parallelization.
    print(f"Loading data from {filename}...")
    try:
        df = pd.read_csv(filename, sep=";", header=0, nrows=100000) 
        # df = pd.read_csv(filename, sep=";", header=0) 
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
        exit()

    # --- 2. Extract Context (Metadata) ---
    # Extract context columns for analysis later. 
    # Assumes these columns are constant per F_SESSION.
    context_cols = ['F_SESSION', 'session', 'k_aircraft', 'k_operator', 'k_mission']
    
    # Filter columns that actually exist in the CSV
    available_cols = [c for c in context_cols if c in df.columns]
    
    # Create context DF (1 row per flight/session)
    df_context = df[available_cols].drop_duplicates(subset=['F_SESSION']).set_index('F_SESSION')
    print(f"Context extracted for {len(df_context)} unique sessions.")

    # --- 3. Build Sequences ---
    df_sequence = process_flight_sequences(df, md=1.0)
    print(f"Sequences constructed. Shape: {df_sequence.shape}")

    # --- 4. Logical Compression ---
    logical_sequences = build_logical_sequences(df_sequence)
    print("Logical compression completed.")

    # --- 5. Similarity Matrix ---
    # Sync indices: Ensure we only process flights present in both sequences and context
    common_indices = logical_sequences.index.intersection(df_context.index)
    
    if len(common_indices) < len(logical_sequences):
        print(f"Warning: Dropped {len(logical_sequences) - len(common_indices)} flights due to missing context.")

    logical_sequences = logical_sequences.loc[common_indices]
    df_context = df_context.loc[common_indices]

    sim_matrix = build_similarity_matrix(logical_sequences)
    print("Similarity matrix calculated.")

    # --- 6. Hierarchical Clustering ---
    # Convert similarity to distance (Distance = 1 - Similarity)
    distance_matrix = 1 - sim_matrix.values
    
    # Fix potential floating point errors (negative small numbers -> 0)
    distance_matrix[distance_matrix < 0] = 0
    
    # Convert to condensed distance matrix (required for linkage if symmetric)
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Compute Linkage Matrix (Z) using Average Linkage (UPGMA)
    linkage_matrix = linkage(condensed_dist, method='average')
    print("Linkage Matrix computed.")

    # --- 7. Visualization ---
    visualize_dendrogram(linkage_matrix, sim_matrix.index)

    # --- 8. Cluster Analysis at Different Cuts ---
    # Define distance thresholds to test. 
    # Distance ranges from 0 (Identical) to 1 (Completely different).
    # 0.2 -> Strict clusters (High similarity)
    # 0.8 -> Loose clusters (Low similarity)
    thresholds_to_test = [0.2, 0.4, 0.6, 0.8]
    
    analyze_clusters_at_thresholds(
        linkage_matrix, 
        sim_matrix.index, 
        df_context, 
        thresholds_to_test
    )