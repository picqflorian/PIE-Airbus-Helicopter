from collections import Counter
import math
import pandas as pd


def get_ngrams(sequence, n):
    """Generates n-grams from a list sequence efficiently."""
    if n == 0:
        return []
    # Using zip to create sliding windows is efficient in Python
    return zip(*[sequence[i:] for i in range(n)])


def find_significant_motifs(flights_data, k, z_threshold=1.96, nb_phases=None):
    """
    Identifies statistically significant k-motifs.
    
    Args:
        flights_data (list of lists): Each inner list is a sequence of flight phase integers.
        k (int): The length of the motif to analyze (e.g., 3).
        z_threshold (float): Z-score cutoff (1.96 for 95% confidence).
        nb_phases (int, optional): Number of unique flight phases (required for k=2).
        
    Returns:
        pd.DataFrame: Table of motifs with Obs/Exp probabilities and Z-scores.
    """
    if k == 2 and nb_phases is None:
        raise ValueError("For k=2, nb_phases must be provided to calculate expected probabilities.")
    
    # 1. Count frequencies for k, k-1, and k-2 patterns
    # We use a single pass over the data to populate all counters
    counts_k = Counter()      # Counts for x_1...x_k
    counts_prefix = Counter() # Counts for x_1...x_{k-1}
    counts_overlap = Counter() # Counts for x_2...x_{k-1} (the middle part)
    
    for flight in flights_data:
        # flight is a list of ints, e.g., [10, 20, 30, 40]
        if len(flight) < k:
            continue

        # Count k-grams (x_1...x_k)
        ngrams = list(get_ngrams(flight, k))
        counts_k.update(ngrams)
            
        # Note: If k=2, p_exp = p_obs (Eq. 5)
        if k > 2:
            # Count (k-1)-grams (used for prefix and suffix)
            ngrams_minus_1 = list(get_ngrams(flight, k - 1))
            counts_prefix.update(ngrams_minus_1)

            # Count (k-2)-grams (used for overlap)
            ngrams_minus_2 = list(get_ngrams(flight, k - 2))
            counts_overlap.update(ngrams_minus_2)
    
    # Track total number of substrings of each length for probability normalization
    total_k = counts_k.total()
    total_prefix = counts_prefix.total()
    total_overlap = counts_overlap.total()
    
    # 2. Calculate Probabilities and Z-scores
    results = []
    
    for motif, count in counts_k.items():
        if k == 2:
            # Observed Probability: p_obs(ABC)
            p_obs = count / total_k

            # Expected Probability: p_exp(AB) = poss(AB) / poss(XX)
            p_exp = 1 / nb_phases**2
            expected_count = (total_k/2) * p_exp  # Total possible pairs is total_k/2

            # Standard Deviation for binomial distribution approx: sqrt(N * p * (1-p))
            sigma = math.sqrt((total_k/2) * p_exp * (1 - p_exp))

        else:
            # Define parts of the motif
            # motif is a tuple like (A, B, C)
            prefix = motif[:-1]      # (A, B)
            suffix = motif[1:]       # (B, C)
            overlap = motif[1:-1]    # (B)

            # Observed Probability: p_obs(ABC)
            p_obs = count / total_k
            
            # Expected Probability calculation 
            # p_exp(ABC) = p_obs(AB) * p_obs(BC) / p_obs(B)
            prob_prefix = counts_prefix[prefix] / total_prefix
            prob_suffix = counts_prefix[suffix] / total_prefix

            prob_overlap = counts_overlap[overlap] / total_overlap
            if prob_overlap == 0:
                continue # Avoid division by zero
            p_exp = (prob_prefix * prob_suffix) / prob_overlap
            
            # Standard Deviation for binomial distribution approx: sqrt(N * p * (1-p))
            expected_count = p_exp * total_k
            sigma = math.sqrt(total_k * p_exp * (1 - p_exp))
        
        # Calculate Z-score
        if sigma == 0:
            z_score = 0 # p_exp is too small (prevent floating point issues)
        else:
            z_score = (count - expected_count) / sigma

        if z_score > z_threshold:
            results.append({
                "motif": motif,
                "count": count,
                "expected_count": expected_count,
                "p_obs": p_obs,
                "p_exp": p_exp,
                "z_score": z_score
            })
            
    # 3. Format Output
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="z_score", ascending=False)
    
    return df
