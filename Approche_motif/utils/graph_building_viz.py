import community as community_louvain
import networkx as nx
import numpy as np
import os
import pandas as pd
from pyvis.network import Network
import sqlite3


def build_motif_edgelist_lazy(flights_generator, significant_motifs_df, k, output_csv, z_threshold=1.96, db_path='temp_edges.db'):
    """
    Constructs the weighted network with minimal RAM usage by offloading 
    counts to a temporary SQLite database.
    
    Args:
        flights_generator: A generator yielding one sequence (list/string) at a time.
        significant_motifs_df: DataFrame containing 'motif' and 'p_obs'.
        k: Motif length.
        output_csv: Path to save the final Edge List.
        z_threshold: Z-score cutoff.
        db_path: Path for temporary SQL database (deleted after use).
    """
    
    # --- 1. Setup & Indexing ---
    print("Setting up indices...")
    valid_motifs = significant_motifs_df['motif'].tolist()
    motif_to_idx = {motif: i for i, motif in enumerate(valid_motifs)}
    
    # Map ID -> Probability for Z-score calc
    # We use a simple list where index matches ID for O(1) access
    prob_array = np.zeros(len(valid_motifs))
    motif_probs = significant_motifs_df.set_index('motif')['p_obs'].to_dict()
    for m, idx in motif_to_idx.items():
        prob_array[idx] = motif_probs[m]
        
    # --- 2. Initialize Temporary Database ---
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create a table to store edge counts.
    # We use (source, target) as primary key to handle summing efficiently.
    c.execute('''
        CREATE TABLE IF NOT EXISTS edge_counts (
            source INTEGER, 
            target INTEGER, 
            count INTEGER,
            PRIMARY KEY (source, target)
        )
    ''')
    # Speed up inserts
    c.execute('PRAGMA synchronous = OFF')
    c.execute('PRAGMA journal_mode = MEMORY') 
    
    # --- 3. Stream Processing & Counting ---
    print("Streaming flights and counting pairs to disk...")
    
    normalization_sum = 0
    buffer = {}  # Small in-memory buffer to reduce disk I/O
    BUFFER_SIZE = 1_000_000  # Number of pairs to hold before flushing to DB
    
    processed_count = 0
    
    for flight in flights_generator:
        l_s = len(flight)
        
        # Aggregate normalization sum (Eq. 6 in paper) 
        if l_s >= 2 * k:
            normalization_sum += (l_s - 2 * k + 1) * (l_s - 2 * k + 2)
        
        if l_s < 2 * k:
            continue
            
        # Identify instances of motifs
        instances = []
        for i in range(len(flight) - k + 1):
            segment = tuple(flight[i : i+k])
            if segment in motif_to_idx:
                instances.append((i, motif_to_idx[segment]))
        
        # Count non-overlapping co-occurrences [cite: 76]
        n_inst = len(instances)
        for i in range(n_inst):
            start_x, id_x = instances[i]
            for j in range(i + 1, n_inst):
                start_y, id_y = instances[j]
                
                # Check non-overlapping constraint
                if start_y >= start_x + k:
                    pair = (id_x, id_y)
                    buffer[pair] = buffer.get(pair, 0) + 1
        
        # Flush buffer to SQL if full
        if len(buffer) >= BUFFER_SIZE:
            _flush_buffer_to_db(c, buffer)
            buffer = {} # Reset buffer
            conn.commit()
            print(f"Processed {processed_count} flights...", end='\r')
            
        processed_count += 1

    # Final flush
    if buffer:
        _flush_buffer_to_db(c, buffer)
        conn.commit()
    
    print(f"\nFinished counting. Normalization Sum: {normalization_sum:.2e}")
    
    # --- 4. Compute Z-Scores and Write CSV ---
    print("Calculating Z-scores and writing to CSV...")
    
    # Prepare CSV header
    with open(output_csv, 'w') as f:
        f.write("Source,Target,Obs_count,Exp_count,Weight\n")
        
        # Stream results from DB so we never load the whole edge list
        # We perform Z-score calculation row-by-row here
        c.execute('SELECT source, target, count FROM edge_counts')
        
        while True:
            # Fetch in chunks to keep memory low
            rows = c.fetchmany(10000)
            if not rows:
                break
            
            # Vectorize this small chunk
            data = np.array(rows)
            sources = data[:, 0].astype(int)
            targets = data[:, 1].astype(int)
            obs_counts = data[:, 2]
            
            # Retrieve probabilities
            p_source = prob_array[sources]
            p_target = prob_array[targets]
            
            # Calculate Expected Counts (Eq. 6) 
            # N_exp = 0.5 * p(X) * p(Y) * NormSum
            expected_counts = 0.5 * p_source * p_target * normalization_sum
            
            # Calculate Sigma (Poisson approx)
            sigma = np.sqrt(expected_counts)
            
            # Calculate Z-score
            with np.errstate(divide='ignore', invalid='ignore'):
                z_scores = (obs_counts - expected_counts) / sigma
                z_scores = np.nan_to_num(z_scores)
            
            # Filter and Write
            mask = z_scores > z_threshold
            
            if np.any(mask):
                valid_indices = np.where(mask)[0]
                for idx in valid_indices:
                    src_name = str(valid_motifs[sources[idx]])
                    tgt_name = str(valid_motifs[targets[idx]])
                    obs_count = obs_counts[idx]
                    exp_count = expected_counts[idx]
                    weight = f"{z_scores[idx]:.4f}"
                    
                    # Write line immediately
                    f.write(f'"{src_name}","{tgt_name}",{obs_count},{exp_count},{weight}\n')

    # Cleanup
    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    print("Done.")

def _flush_buffer_to_db(cursor, buffer_dict):
    """
    Helper to upsert counts into SQLite.
    UPSERT syntax (ON CONFLICT DO UPDATE) requires SQLite 3.24+.
    For older python versions, we use standard INSERT with grouping later or helper logic.
    Here assumes modern SQLite or standard replacement logic.
    """
    # Convert dict items to list of tuples for executemany
    # items: [(source, target, count), ...]
    data = [(k[0], k[1], v) for k, v in buffer_dict.items()]
    
    # Upsert Query
    query = '''
        INSERT INTO edge_counts (source, target, count) 
        VALUES (?, ?, ?) 
        ON CONFLICT(source, target) 
        DO UPDATE SET count = count + excluded.count
    '''
    cursor.executemany(query, data)


def flight_data_stream(flights_list):
    for flight in flights_list:
        yield flight


def visualize_top_edges(csv_path, top_n=1000):
    """
    Loads the graph but keeps only the top N strongest edges for clear visualization.
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Sort by Weight (Z-score) descending and take top N
    df_top = df.sort_values(by='Weight', ascending=False).head(top_n)
    df_top.reset_index(drop=True, inplace=True)
    df_top.drop(columns=['Obs_count', 'Exp_count'], inplace=True)
    
    # Map tuples to indices for visualization
    edges = set(df_top['Source']).union(set(df_top['Target']))

    edge_to_idx = {edge: i for i, edge in enumerate(edges)}
    idx_to_edge = {i: edge for i, edge in enumerate(edges)}

    df_top['Source'] = df_top['Source'].map(edge_to_idx)
    df_top['Target'] = df_top['Target'].map(edge_to_idx)

    # Create smaller graph for visualization
    G_viz = nx.from_pandas_edgelist(
        df_top, 
        source='Source', 
        target='Target', 
        edge_attr='Weight', 
        create_using=nx.DiGraph
    )
    
    print(f"Visualizing top {top_n} edges out of {len(df)} total significant edges.")
    return G_viz, idx_to_edge


def create_interactive_graph(G, filename="html-graphs/motif_network.html"):
    # Initialize PyVis network
    net = Network(height="750px", width="100%", notebook=False, cdn_resources='remote', directed=True)
    
    # Detect Communities (Louvain method) to color nodes
    G = detect_communities(G)

    # Convert NetworkX graph to PyVis
    net.from_nx(G)

    # Optional: Add physics controls (makes it fun to play with)
    net.show_buttons(filter_=['physics'])
    
    # Save the HTML file
    # Note: net.show() writes the file and attempts to display it, 
    # but IFrame is often more reliable for inline viewing.
    net.write_html(filename)


def detect_communities(G):
    """ Detects communities in the graph using the Louvain method."""
    try:
        partition = community_louvain.best_partition(G.to_undirected())
        # Add partition info to node attributes for coloring
        for node, group_id in partition.items():
            G.nodes[node]['group'] = group_id
    except ImportError:
        print("Community detection skipped (install 'python-louvain' for colors)")
    
    return G