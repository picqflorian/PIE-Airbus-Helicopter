import argparse
import json
import numpy as np
import pandas as pd
from utils.motif_finding import find_significant_motifs

def main():
    # 1. Set up argument parsing for terminal inputs
    parser = argparse.ArgumentParser(description="Find significant motifs in flight data.")
    parser.add_argument("-f", "--filename", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--md", type=float, default=1.0, help="Duration step (md)")
    parser.add_argument("-k", "--k", type=int, default=5, help="Length of motifs (k)")
    parser.add_argument("--min_count", type=int, default=2, help="Minimum count for a motif to be kept")
    
    args = parser.parse_args()

    # Assign inputs
    filename = args.filename
    md = args.md
    k = args.k
    min_count = args.min_count

    print(f"Loading data from {filename}...")
    
    # 2. Data processing
    df = pd.read_csv(filename, sep=";", header=0, index_col=0)
    df.drop(columns=['session'], inplace=True)
    df = df.sort_values(by=['F_SESSION', 'F_START_FRAME'], ascending=[True, True])

    first = df['FIRST_WORD_INDEX']
    second = df['SECOND_WORD_INDEX']
    third = df['THIRD_WORD_INDEX']

    df['phase'] = list(zip(first, second, third))
    phase_to_idx = {phase: i for i, phase in enumerate(df['phase'].unique())}
    idx_to_phase = {i: phase for i, phase in enumerate(df['phase'].unique())}
    df['phase_idx'] = df['phase'].map(phase_to_idx)

    df.drop(columns=['phase', 'FIRST_WORD_INDEX', 'SECOND_WORD_INDEX', 'THIRD_WORD_INDEX', 'F_START_FRAME', 'F_END_FRAME'], inplace=True)

    flight_info = df.groupby('F_SESSION')[["k_aircraft", "k_operator", "k_mission"]].max().reset_index()
    sessions = df['F_SESSION'].unique()

    df['F_DURATION'] = pd.to_timedelta(df['F_DURATION'])
    df['duration_sec'] = df['F_DURATION'].dt.total_seconds()
    df['reps'] = np.ceil(df['duration_sec'] / md).astype(int)

    df.drop(columns=['F_DURATION', 'duration_sec'], inplace=True)
    df = df.loc[df.index.repeat(df['reps'])].copy()
    df.drop(columns=['reps'], inplace=True)

    flights = [df[df['F_SESSION'] == s]['phase_idx'].to_list() for s in sessions]

    # 3. Significant motif finding
    print(f"Finding significant motifs (k={k})...")
    df_significant = find_significant_motifs(flights, k, z_threshold=1.96, nb_phases=None)
    
    # Filter and sort
    df_significant = df_significant[df_significant['count'] > min_count]
    df_significant = df_significant.sort_values(by="z_score", ascending=False)
    
    print(f"Total significant motifs found (count > {min_count}): {len(df_significant)}")

    # 4. Output results to CSV
    # Convert the idx_to_phase dictionary to a DataFrame to easily save it as CSV
    df_idx_to_phase = pd.DataFrame(list(idx_to_phase.items()), columns=['idx', 'phase'])
    
    df_idx_to_phase.to_csv("idx_to_phase.csv", index=False)
    df_significant.to_csv("significant_motifs.csv", index=False)
    flight_info.to_csv("flight_info.csv", index=False)

    # Export the flights list as a JSON artifact
    with open("flights.json", "w") as f:
        json.dump(flights, f)
    
    print("Execution complete. Saved 'flights.json', 'idx_to_phase.csv', 'flight_info.csv' and 'significant_motifs.csv' to the current directory.")

if __name__ == "__main__":
    main()