# ============================================================
# 0. Imports & Options
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    df = df.copy()
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
    
    df.loc[df['reps'] < 1, 'reps'] = 1
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
    return [(phase, len(list(group))) for phase, group in groupby(sequence)]


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
    """
    flights = logical_sequences.index.tolist()
    n = len(flights)
    sim_matrix = np.zeros((n, n))
    
    print(f"Calculating similarity matrix for {n} flights...")

    for i in range(n):
        # if i % 10 == 0: 
        #     print(f"Processing row {i}/{n}")
                
        print(f"Processing row {i}/{n}")
            
        for j in range(i, n): 
            print(f"-------------- i = {i}, j = {j}") 
            score = weighted_lcs_similarity(
                logical_sequences.iloc[i],
                logical_sequences.iloc[j]
            )
            sim_matrix[i, j] = score
            sim_matrix[j, i] = score

    return pd.DataFrame(sim_matrix, index=flights, columns=flights)


# ============================================================
# 5. Visualization & Cluster Analysis (SAVING TO FILES)
# ============================================================

def visualize_dendrogram(linkage_matrix, labels):
    """
    Plots the standard Hierarchical Clustering Dendrogram.
    """
    plt.figure(figsize=(12, 5))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.ylabel("Distance (1 - Similarity)")
    plt.xlabel("Flights (F_SESSION)")
    plt.tight_layout()
    plt.show()

def visualize_dendrogram_with_thresholds(linkage_matrix, labels, thresholds):
    """
    Plots the Hierarchical Clustering Dendrogram with horizontal lines 
    representing the cut thresholds used for cluster generation.
    """
    plt.figure(figsize=(12, 5))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    
    # Define some colors for the threshold lines
    colors = ['r', 'g', 'm', 'c', 'orange', 'purple']
    
    # Plot a horizontal line for each threshold
    for i, t in enumerate(thresholds):
        color = colors[i % len(colors)]
        plt.axhline(y=t, color=color, linestyle='--', linewidth=1.5, label=f'Cut = {t}')
        
    plt.title("Hierarchical Clustering Dendrogram with Cut Thresholds")
    plt.ylabel("Distance (1 - Similarity)")
    plt.xlabel("Flights (F_SESSION)")
    
    # Place legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def analyze_clusters_at_thresholds(linkage_matrix, sim_matrix_index, df_context, thresholds, 
                                   report_filename="clustering_report.txt", 
                                   assignments_filename="cluster_assignments.csv"):
    """
    Cuts the dendrogram at specified thresholds, prints the results, 
    and saves the crosstabs and raw assignments to files.
    """
    
    # This dataframe will store the cluster ID for every flight at every threshold
    df_assignments = pd.DataFrame(index=sim_matrix_index)
    
    # Open text file to write the report
    with open(report_filename, "w") as f:
        f.write("=== HIERARCHICAL CLUSTERING REPORT ===\n")
        
        for t in thresholds:
            header = f"\n{'='*80}\nCUTTING DENDROGRAM AT THRESHOLD (DISTANCE) = {t}\n{'='*80}\n"
            print(header)
            f.write(header)
            
            # Get cluster labels
            cluster_labels = fcluster(linkage_matrix, t=t, criterion='distance')
            
            # Save labels to our assignments dataframe
            df_assignments[f'Cluster_Dist_{t}'] = cluster_labels
            
            # Create a temporary dataframe for analysis
            res = pd.DataFrame(index=sim_matrix_index)
            res['Cluster'] = cluster_labels
            res = res.join(df_context, how='left')
            
            num_clusters = res['Cluster'].nunique()
            msg_clusters = f"-> Number of clusters generated: {num_clusters}\n"
            print(msg_clusters)
            f.write(msg_clusters)
            
            # 1. Cluster Sizes
            msg_sizes = "\n--- Cluster Sizes ---\n"
            print(msg_sizes)
            f.write(msg_sizes)
            
            sizes_str = res['Cluster'].value_counts().sort_index().to_string()
            print(sizes_str)
            f.write(sizes_str + "\n")
            
            # 2. Crosstabs (Relationship with Metadata)
            for col in ['k_aircraft', 'k_mission', 'k_operator']:
                if col in res.columns:
                    msg_cross = f"\n--- {col.replace('k_', '').capitalize()} Distribution per Cluster ---\n"
                    print(msg_cross)
                    f.write(msg_cross)
                    
                    ct = pd.crosstab(res['Cluster'], res[col])
                    print(ct)
                    f.write(ct.to_string() + "\n")
                    
    # Merge assignments with metadata and save to CSV
    df_final_assignments = df_assignments.join(df_context, how='left')
    df_final_assignments.to_csv(assignments_filename, sep=";")
    
    success_msg = f"\n[SUCCESS] Report saved to: {report_filename}\n[SUCCESS] Raw assignments saved to: {assignments_filename}"
    print(success_msg)


# ============================================================
# 6. Main Execution Pipeline
# ============================================================

if __name__ == "__main__":

    # --- 1. Load Data ---
    filename = "PIE_data_with_context.csv"
    
    print(f"Loading data from {filename}...")
    try:
        df = pd.read_csv(filename, sep=";", header=0, nrows=150000) 
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
        exit()

    # --- 2. Extract Context (Metadata) ---
    context_cols = ['F_SESSION', 'session', 'k_aircraft', 'k_operator', 'k_mission']
    available_cols = [c for c in context_cols if c in df.columns]
    
    df_context = df[available_cols].drop_duplicates(subset=['F_SESSION']).set_index('F_SESSION')
    print(f"Context extracted for {len(df_context)} unique sessions.")

    # --- 3. Build Sequences ---
    df_sequence = process_flight_sequences(df, md=1.0)
    print(f"Sequences constructed. Shape: {df_sequence.shape}")

    # --- 4. Logical Compression ---
    logical_sequences = build_logical_sequences(df_sequence)
    print("Logical compression completed.")

    # --- 5. Similarity Matrix ---
    common_indices = logical_sequences.index.intersection(df_context.index)
    logical_sequences = logical_sequences.loc[common_indices]
    df_context = df_context.loc[common_indices]

    sim_matrix = build_similarity_matrix(logical_sequences)
    print("Similarity matrix calculated.")

    # --- 6. Hierarchical Clustering ---
    distance_matrix = 1 - sim_matrix.values
    distance_matrix[distance_matrix < 0] = 0
    condensed_dist = squareform(distance_matrix, checks=False)
    
    linkage_matrix = linkage(condensed_dist, method='average')
    print("Linkage Matrix computed.")

    # --- 7. Thresholds Definition ---
    thresholds_to_test = [0.2, 0.4, 0.6, 0.8]

    # --- 8. Visualizations ---
    # Plot standard dendrogram (Original)
    visualize_dendrogram(linkage_matrix, sim_matrix.index)
    
    # Plot new dendrogram with horizontal threshold lines
    visualize_dendrogram_with_thresholds(linkage_matrix, sim_matrix.index, thresholds_to_test)

    # --- 9. Cluster Analysis & File Export ---
    analyze_clusters_at_thresholds(
        linkage_matrix, 
        sim_matrix.index, 
        df_context, 
        thresholds_to_test,
        report_filename="LCS_clustering_report.txt",
        assignments_filename="LCS_cluster_assignments.csv"
    )