import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

def process_flight_sequences(df, md=1.0):
    df = df.copy()
    
    # Tri indispensable pour respecter la chronologie
    df = df.sort_values(by=['F_SESSION', 'F_START_FRAME'], ascending=[True, True])

    # 1. Formatage des Index (008, 023, etc.) et Création ID
    target_cols = ['FIRST_WORD_INDEX', 'SECOND_WORD_INDEX', 'THIRD_WORD_INDEX']
    
    # Pour chaque colonne d'index, on convertit en int, puis str, puis on ajoute les zéros
    for col in target_cols:
        if col == 'THIRD_WORD_INDEX':
            # Le troisième index peut aller jusqu'à 4 chiffres
            df[col] = df[col].astype(int).astype(str).str.zfill(4)
        else:
            df[col] = df[col].astype(int).astype(str).str.zfill(3)
        
    # Concaténation : 003 + 010 + 285 = "003010285" (toujours 9 caractères)
    df['Phase_ID'] = df[target_cols].agg(''.join, axis=1)

    # 2. Gestion de la répétition (md ou sequence brute)
    if md is not None:
        # Mode Temporel : on calcule selon la durée
        df['F_DURATION'] = pd.to_timedelta(df['F_DURATION'])
        df['duration_sec'] = df['F_DURATION'].dt.total_seconds()
        # Arrondi inférieur
        df['reps'] = np.floor(df['duration_sec'] / md).astype(int)
    else:
        # Mode Séquentiel pur : on garde la ligne telle quelle (1 répétition)
        df['reps'] = 1

    # 3. Expansion
    # On ne garde que les lignes où reps > 0 (pour nettoyer les durées < md)
    df = df[df['reps'] > 0]
    df_expanded = df.loc[df.index.repeat(df['reps'])].copy()
    
    # 4. Compteur de pas de temps (ou d'étape)
    df_expanded['step_count'] = df_expanded.groupby('F_SESSION').cumcount()
    
    # 5. Pivot
    df_sequence = df_expanded.pivot(
        index='F_SESSION', 
        columns='step_count', 
        values='Phase_ID'
    )
    
    return df_sequence

# --- Test ---
df = pd.read_csv("PIE_data.csv", sep=";", header=0, index_col=0)
df = df.sort_values(by=['F_SESSION', 'F_START_FRAME'], ascending=[True, True])

print("--- df utilisé, les 4000 premières lignes ---")
test = df.head(4000) # On prend un échantillon
print(test.head())

# On va garder que les 10 premières colonnes des 10 premièrs vols pour l'affichage
print("--- Test avec md = 1 (Temporel) ---")
res_time = process_flight_sequences(df, md=1)
print(res_time.iloc[:, :10].head(10)) 

print("\n--- Test avec md = None (Séquentiel pur) ---")
res_seq = process_flight_sequences(df, md=None)
print(res_seq.iloc[:, :10].head(10))


