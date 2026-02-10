# ============================================================
# 0. Imports & Options
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import groupby
from scipy.cluster.hierarchy import linkage, dendrogram

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

    df = df.copy()

    # Chronological order
    df = df.sort_values(by=['F_SESSION', 'F_START_FRAME'])

    # --- Phase ID construction ---
    df['FIRST_WORD_INDEX']  = df['FIRST_WORD_INDEX'].astype(int).astype(str).str.zfill(3)
    df['SECOND_WORD_INDEX'] = df['SECOND_WORD_INDEX'].astype(int).astype(str).str.zfill(3)
    df['THIRD_WORD_INDEX']  = df['THIRD_WORD_INDEX'].astype(int).astype(str).str.zfill(4)

    df['Phase_ID'] = (
        df['FIRST_WORD_INDEX'] +
        df['SECOND_WORD_INDEX'] +
        df['THIRD_WORD_INDEX']
    )

    # --- Duration handling ---
    df['F_DURATION'] = pd.to_timedelta(df['F_DURATION'])
    df['duration_sec'] = df['F_DURATION'].dt.total_seconds()

    if md is not None:
        df['reps'] = np.floor(df['duration_sec'] / md).astype(int)
    else:
        df['reps'] = 1

    df = df[df['reps'] > 0]

    # --- Temporal expansion ---
    df_expanded = df.loc[df.index.repeat(df['reps'])].copy()
    df_expanded['step'] = df_expanded.groupby('F_SESSION').cumcount()

    # --- Final sequence table ---
    df_sequence = df_expanded.pivot(
        index='F_SESSION',
        columns='step',
        values='Phase_ID'
    )

    return df_sequence


# ============================================================
# 2. Logical Waypoints with Duration
# ============================================================

def compress_sequence_with_duration(sequence):
    """
    Compresses a symbolic sequence into (Phase_ID, duration) tuples.
    """
    return [(phase, len(list(group)))
            for phase, group in groupby(sequence)]


def build_logical_sequences(df_sequence):
    """
    Applies logical compression to all flights.
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
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            p1, d1 = seq1[i]
            p2, d2 = seq2[j]

            if p1 == p2:
                dp[i + 1][j + 1] = dp[i][j] + min(d1, d2)
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[m][n]


def weighted_lcs_similarity(seq1, seq2):
    """
    Normalized similarity score in [0, 1].
    """
    total_duration = min(
        sum(d for _, d in seq1),
        sum(d for _, d in seq2)
    )

    if total_duration == 0:
        return 0.0

    return weighted_lcs(seq1, seq2) / total_duration


# ============================================================
# 4. Similarity Matrix
# ============================================================

def build_similarity_matrix(logical_sequences):
    """
    Computes pairwise weighted LCS similarity matrix.
    """
    flights = logical_sequences.index.tolist()
    n = len(flights)

    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            print(f"i = {i}, j = {j}")
            sim_matrix[i, j] = weighted_lcs_similarity(
                logical_sequences.iloc[i],
                logical_sequences.iloc[j]
            )

    return pd.DataFrame(sim_matrix, index=flights, columns=flights)


# ============================================================
# 5. Visualization (Slides-Ready)
# ============================================================

def visualize_results(sim_matrix):
    """
    Heatmap + Hierarchical clustering dendrogram.
    """

    # --- Similarity Heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="viridis")
    plt.title("Weighted LCS Similarity Between Flights")
    plt.tight_layout()
    plt.show()

    # --- Hierarchical Clustering ---
    distance_matrix = 1 - sim_matrix.values
    linkage_matrix = linkage(distance_matrix, method='average')

    plt.figure(figsize=(12, 5))
    dendrogram(
        linkage_matrix,
        labels=sim_matrix.index,
        leaf_rotation=90
    )
    plt.title("Hierarchical Clustering of Flight Sequences")
    plt.ylabel("1 - Similarity")
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Full Pipeline Execution
# ============================================================

if __name__ == "__main__":

    # --- Load data ---
    # df = pd.read_csv("PIE_data.csv", sep=";", header=0)
    df = pd.read_csv("PIE_data.csv", sep=";", header=0, nrows=30000)
    print("LOAD DATA")

    # --- Build sequences ---
    df_sequence = process_flight_sequences(df, md=1.0)
    print("BUILD SEQUENCES")

    # --- Logical compression ---
    logical_sequences = build_logical_sequences(df_sequence)
    print("LOGICAL COMPRESSION")

    # --- Similarity matrix ---
    sim_matrix = build_similarity_matrix(logical_sequences)
    print("SIMILARITY MATRIX")

    # --- Visualization ---
    visualize_results(sim_matrix)
    print("VISUALIZATION")
