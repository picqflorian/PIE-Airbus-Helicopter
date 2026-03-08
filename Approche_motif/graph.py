import argparse
import json
import pandas as pd
from utils.graph_building_viz import (
    build_motif_edgelist_lazy, 
    flight_data_stream, 
    visualize_top_edges, 
    create_interactive_graph
)

def main():
    # 1. Set up argument parsing
    parser = argparse.ArgumentParser(description="Build and visualize a motif network graph.")
    
    # Core inputs for this script
    parser.add_argument("-k", "--k", type=int, required=True, help="Length of motifs (k). Must match the k used in extraction.")
    parser.add_argument("--html_out", type=str, default="html-graphs/motif_network.html", help="Filepath for the output HTML graph")
    parser.add_argument("--top_edges", type=int, default=2000, help="Number of top edges to visualize")
    
    # File handling inputs (defaulted to the outputs of the first script)
    parser.add_argument("--flights_file", type=str, default="flights.json", help="Path to the processed flights JSON file")
    parser.add_argument("--motifs_file", type=str, default="significant_motifs.csv", help="Path to the significant motifs CSV file")
    parser.add_argument("--network_csv", type=str, default="final_network.csv", help="Filepath to save the intermediate network CSV")

    args = parser.parse_args()

    print(f"Loading preprocessed flights from '{args.flights_file}'...")
    with open(args.flights_file, "r") as f:
        flights = json.load(f)

    print(f"Loading significant motifs from '{args.motifs_file}'...")
    df_significant = pd.read_csv(args.motifs_file)

    # 2. Build the motif edgelist
    print("Building the motif edgelist. This might take a moment...")
    build_motif_edgelist_lazy(
        flights_generator=flight_data_stream(flights), 
        significant_motifs_df=df_significant, 
        k=args.k, 
        output_csv=args.network_csv
    )

    # 3. Analyze the generated network
    df = pd.read_csv(args.network_csv)
    print("\n--- Network Statistics ---")
    print(f"Total edges in final network: {len(df)}")
    print(f"Total nodes in final network: {len(set(df['Source']).union(set(df['Target'])))}")

    # 4. Visualize and generate HTML
    print(f"\nExtracting top {args.top_edges} edges for visualization...")
    G, idx_to_edge = visualize_top_edges(args.network_csv, top_n=args.top_edges)
    
    print(f"Visualization Graph Stats -> Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print(f"Generating interactive HTML graph at '{args.html_out}'...")
    create_interactive_graph(G, filename=args.html_out)
    
    print("Done! Your interactive graph is ready.")

if __name__ == "__main__":
    main()