import argparse
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.motif_finding import get_ngrams

def main():
    # 1. Set up argument parsing
    parser = argparse.ArgumentParser(description="Cluster flights based on a bag-of-words representation of motifs.")
    
    # Core inputs
    parser.add_argument("-k", "--k", type=int, default=5, help="Length of motifs (k). Must match previous scripts.")
    parser.add_argument("-c", "--n_clusters", type=int, default=4, help="Number of clusters to generate (chosen_k)")
    
    # File handling inputs (defaulting to the pipeline's generated names)
    parser.add_argument("--flight_info", type=str, default="flight_info.csv", help="Path to the flight info CSV")
    parser.add_argument("--motifs_file", type=str, default="significant_motifs.csv", help="Path to the significant motifs CSV")
    parser.add_argument("--flights_file", type=str, default="flights.json", help="Path to the processed flights JSON")
    parser.add_argument("--output_csv", type=str, default="flight_clusters.csv", help="Filepath to save the final clustering results")

    args = parser.parse_args()

    # 2. Load the data
    print("Loading data artifacts...")
    flight_info = pd.read_csv(args.flight_info)
    df_significant = pd.read_csv(args.motifs_file)
    
    with open(args.flights_file, "r") as f:
        flights = json.load(f)

    # 3. Process into a Bag of Words corpus
    print("Building flight document corpus...")
    rows_list = []
    significant_motifs = set(df_significant["motif"])
    sessions = flight_info['F_SESSION'].unique()

    for i, flight in enumerate(flights):
        # flight is a list of ints, e.g., [10, 20, 30, 40]
        if len(flight) < args.k:
            continue

        # Get significant k-grams from the flight
        ngrams = pd.Series(get_ngrams(flight, args.k))
        
        # Note: If get_ngrams returns tuples but df_significant["motif"] contains strings,
        # you might need to format ngrams to strings BEFORE this .isin() check depending on your exact data types.
        ngrams = ngrams[ngrams.isin(significant_motifs)]
        ngrams = ngrams.apply(lambda x: '_'.join(map(str, x))) # Transform k-grams (x1_x2_xk)

        ngrams = " ".join(ngrams.astype(str))
        
        rows_list.append({'flight_id': sessions[i], 'flight_document': ngrams})

    flight_corpus = pd.DataFrame(rows_list)

    # 4. Vectorize (TF-IDF)
    print("Vectorizing documents using TF-IDF...")
    tfidf = TfidfVectorizer(
        lowercase=False,  # Motifs aren't case sensitive
        token_pattern=r"(?u)\b\w+\b" # Default pattern works, but ensuring it captures numbers/underscores
    )

    X = tfidf.fit_transform(flight_corpus['flight_document'])

    # 5. Final Clustering
    print(f"Running KMeans clustering (k={args.n_clusters})...")
    final_kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init='auto')
    flight_corpus['cluster'] = final_kmeans.fit_predict(X)

    # 6. View & Save results
    print(f"\nFlights clustered into {args.n_clusters} groups:")
    print(flight_corpus[['flight_id', 'cluster']].head(10))

    for i in range(args.n_clusters):
        ids = flight_corpus[flight_corpus["cluster"] == i]["flight_id"]
        aux = flight_info[flight_info["F_SESSION"].isin(set(ids))]

        aircraft_per = len(aux["k_aircraft"].unique()) / len(flight_info["k_aircraft"].unique()) * 100
        operator_per = len(aux["k_operator"].unique()) / len(flight_info["k_operator"].unique()) * 100
        mission_per = len(aux["k_mission"].unique()) / len(flight_info["k_mission"].unique()) * 100

        print(f"Cluster nb {i} had:")
        print(f"\t{aircraft_per:.2f} % of the {len(flight_info['k_aircraft'].unique())} aircrafts")
        print(f"\t{operator_per:.2f} % of the {len(flight_info['k_operator'].unique())} operators")
        print(f"\t{mission_per:.2f} % of the {len(flight_info['k_mission'].unique())} missions\n")

    # Create a DataFrame of the cluster centers
    feature_names = tfidf.get_feature_names_out()
    ordered_centroids = final_kmeans.cluster_centers_.argsort()[:, ::-1]

    print("\nTop motifs per cluster:")
    # FIXED: Replaced range(k) with range(args.n_clusters)
    for i in range(args.n_clusters):
        print(f"Cluster {i}:")
        # Print the top 3 motifs for this cluster
        for ind in ordered_centroids[i, :3]: 
            print(f" - Motif {feature_names[ind]} (Weight: {final_kmeans.cluster_centers_[i, ind]:.3f})")

    # Save to CSV
    print(f"\nSaving clustering results to '{args.output_csv}'...")
    flight_corpus[['flight_id', 'cluster']].to_csv(args.output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    main()