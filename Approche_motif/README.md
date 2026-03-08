

|       | Seg | Ter | Qua |
|-------|-----|-----|-----|
| Manha | Brainstorming features a criar | Feature engineering | Training/ Explainability |
| Tarde | Feature engineering | Feature engineering / Training  | Final touches |


Here is a concise `README.md` that you can add directly to your project repository. It covers the setup, usage, parameters, and the specific data format your script expects.

```md
# Flight Phase Motif Extractor

This script processes flight context data to identify significant phase motifs. It reads a dataset of flight events, discretizes them based on duration, and extracts repeating sequence patterns (motifs) that occur more frequently than expected by chance.

## Requirements

Ensure you have the following dependencies installed in your Python environment:
* `pandas`
* `numpy`
* A local `utils.motif_finding` module containing the `find_significant_motifs` function.

## Usage

Run the script from your terminal using the following command structure:

```bash
python motif_extractor.py --filename path/to/your/data.csv --md 1.0 -k 5 --min_count 2

```

### Command-Line Arguments

| Argument | Short | Default | Description |
| --- | --- | --- | --- |
| `--filename` | `-f` | **Required** | The path to the input CSV file containing your flight data. |
| `--md` |  | `1.0` | **Duration step (seconds):** Used to discretize the data. The script calculates repetitions by dividing the duration in seconds by this value. |
| `--k` | `-k` | `5` | **Motif length:** The sequence length (number of consecutive phases) to look for when identifying motifs. |
| `--min_count` |  | `2` | **Minimum count:** The minimum number of times a significant motif must appear in the data to be saved in the final output. |

## Input Data Format

The script expects a **semicolon-separated (`;`)** CSV file containing a header row and an index column.

The dataset must contain the following columns and format:

|  | F_SESSION | F_START_FRAME | F_END_FRAME | F_DURATION | FIRST_WORD_INDEX | SECOND_WORD_INDEX | THIRD_WORD_INDEX | session | k_aircraft | k_operator | k_mission |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 3152409 | 742 | 789 | 00:00:24.000 | 3.0 | 10.0 | 285.0 | 3152409 | 41622 | 81 | 1 |
| **1** | 3152409 | 790 | 855 | 00:00:33.000 | 3.0 | 75.0 | 281.0 | 3152409 | 41622 | 81 | 1 |
| **2** | 3152409 | 856 | 965 | 00:00:55.000 | 3.0 | 10.0 | 8.0 | 3152409 | 41622 | 81 | 1 |
| **3** | 3152409 | 966 | 967 | 00:00:01.000 | 3.0 | 10.0 | 155.0 | 3152409 | 41622 | 81 | 1 |
| **4** | 3152409 | 968 | 1001 | 00:00:17.000 | 3.0 | 75.0 | 146.0 | 3152409 | 41622 | 81 | 1 |

*Note: The script automatically drops the `session` column and reconstructs the phase sequences using the `FIRST`, `SECOND`, and `THIRD_WORD_INDEX` columns.*

## Outputs

After a successful run, the script generates two files in your current working directory:

1. `idx_to_phase.csv`: A mapping dictionary bridging the generated numeric phase IDs back to the original `(first, second, third)` word index tuples.
2. `significant_motifs.csv`: A dataframe of all found motifs meeting the significance threshold (`z_score >= 1.96`) and the `--min_count` parameter, sorted by significance.

```

---

Would you like me to also write a `requirements.txt` file or a batch/bash script to help you automate running multiple experiments back-to-back?

```



Here is the updated `README.md`. I have restructured it slightly to present the project as a two-step data pipeline, making it very clear how the outputs of the first script feed directly into the second.

```md
# Flight Phase Motif Pipeline

This project contains a two-step data pipeline to process flight context data, extract significant repeating phase motifs, and visualize them as an interactive network graph.

## Requirements

Ensure you have the following dependencies installed in your Python environment:
* `pandas`
* `numpy`
* Local utility modules: `utils.motif_finding` and `utils.graph_building_viz`

## Pipeline Workflow

The analysis is broken down into two scripts that should be run sequentially. The first script processes the raw data and extracts the motifs, saving intermediate artifacts that the second script uses to build the visualization.

### Step 1: Motif Extraction (`motif_extractor.py`)

This script reads the raw flight events, discretizes them based on duration, and extracts repeating sequence patterns (motifs) that occur more frequently than expected by chance.

**Usage:**
```bash
python motif_extractor.py -f path/to/your/data.csv --md 1.0 -k 5 --min_count 2

```

**Command-Line Arguments:**
| Argument | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--filename` | `-f` | **Required** | The path to the input CSV file containing your flight data. |
| `--md` | | `1.0` | **Duration step (seconds):** Used to discretize the data. |
| `--k` | `-k` | `5` | **Motif length:** The sequence length (number of consecutive phases). |
| `--min_count`| | `2` | **Minimum count:** The minimum number of times a significant motif must appear. |

**Outputs:**

1. `idx_to_phase.csv`: Mapping dictionary of numeric phase IDs to original word index tuples.
2. `significant_motifs.csv`: A dataframe of all found motifs meeting the significance threshold.
3. `flights.json`: A processed list-of-lists containing the phase sequence for each flight (used by Step 2).

---

### Step 2: Network Visualization (`network_builder.py`)

This script takes the extracted motifs and the processed flight sequences to build a directed graph. The nodes represent significant motifs, and the edges are weighted by how often one motif directly follows another in the flight data.

**Usage:**

```bash
python network_builder.py -k 5 --top_edges 1500 --html_out "html-graphs/my_network.html"

```

**Command-Line Arguments:**
| Argument | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--k` | `-k` | **Required** | **Motif length:** Must perfectly match the `-k` used in Step 1. |
| `--html_out` | | `html-graphs/motif_network.html` | Filepath for the generated interactive HTML graph. |
| `--top_edges`| | `2000` | The maximum number of strongest edges to include in the visualization. |
| `--flights_file`| | `flights.json` | Path to the flights artifact generated in Step 1. |
| `--motifs_file`| | `significant_motifs.csv` | Path to the motifs artifact generated in Step 1. |
| `--network_csv`| | `final_network.csv` | Path to save the intermediate raw edge list. |

**Outputs:**

1. `final_network.csv`: The complete calculated edgelist showing source motifs, target motifs, and their transition weights.
2. An interactive HTML graph saved to your specified `--html_out` location.

---

## Input Data Format (For Step 1)

The initial script expects a **semicolon-separated (`;`)** CSV file containing a header row and an index column.

The dataset must contain the following columns and format:

|  | F_SESSION | F_START_FRAME | F_END_FRAME | F_DURATION | FIRST_WORD_INDEX | SECOND_WORD_INDEX | THIRD_WORD_INDEX | session | k_aircraft | k_operator | k_mission |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 3152409 | 742 | 789 | 00:00:24.000 | 3.0 | 10.0 | 285.0 | 3152409 | 41622 | 81 | 1 |
| **1** | 3152409 | 790 | 855 | 00:00:33.000 | 3.0 | 75.0 | 281.0 | 3152409 | 41622 | 81 | 1 |
| **2** | 3152409 | 856 | 965 | 00:00:55.000 | 3.0 | 10.0 | 8.0 | 3152409 | 41622 | 81 | 1 |
| **3** | 3152409 | 966 | 967 | 00:00:01.000 | 3.0 | 10.0 | 155.0 | 3152409 | 41622 | 81 | 1 |
| **4** | 3152409 | 968 | 1001 | 00:00:17.000 | 3.0 | 75.0 | 146.0 | 3152409 | 41622 | 81 | 1 |

*Note: The script automatically drops the `session` column and reconstructs the phase sequences using the `FIRST`, `SECOND`, and `THIRD_WORD_INDEX` columns.*

```

Would you like me to create a simple `run_pipeline.sh` bash script so you can execute both Python files back-to-back with a single terminal command?

```

Here is the complete, finalized `README.md` for your project. I've updated it to reflect the full three-step architecture, added `scikit-learn` to the dependencies, and included the new clustering inputs and outputs.

I also made sure to list `flight_info.csv` as an output of Step 1, so the data flow is completely transparent to anyone reading your documentation.

```md
# Flight Phase Motif Pipeline

This project contains a three-step data pipeline to process flight context data, extract significant repeating phase motifs, visualize them as an interactive network graph, and cluster flights based on their motif profiles using a Bag-of-Words approach.

## Requirements

Ensure you have the following dependencies installed in your Python environment:
* `pandas`
* `numpy`
* `scikit-learn`
* Local utility modules: `utils.motif_finding` and `utils.graph_building_viz`

## Pipeline Workflow

The analysis is broken down into three scripts that should be run sequentially. The intermediate artifacts generated by each step are automatically ingested by the next.

### Step 1: Motif Extraction (`motif_extractor.py`)

This script reads the raw flight events, discretizes them based on duration, and extracts repeating sequence patterns (motifs) that occur more frequently than expected by chance.

**Usage:**
```bash
python motif_extractor.py -f path/to/your/data.csv --md 1.0 -k 5 --min_count 2

```

**Command-Line Arguments:**
| Argument | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--filename` | `-f` | **Required** | The path to the input CSV file containing your flight data. |
| `--md` | | `1.0` | **Duration step (seconds):** Used to discretize the data. |
| `--k` | `-k` | `5` | **Motif length:** The sequence length (number of consecutive phases). |
| `--min_count`| | `2` | **Minimum count:** The minimum number of times a significant motif must appear. |

**Outputs:**

1. `idx_to_phase.csv`: Mapping dictionary of numeric phase IDs to original word index tuples.
2. `significant_motifs.csv`: A dataframe of all found motifs meeting the significance threshold.
3. `flight_info.csv`: Extracted metadata for each flight session.
4. `flights.json`: A processed list-of-lists containing the phase sequence for each flight.

---

### Step 2: Network Visualization (`network_builder.py`)

This script takes the extracted motifs and the processed flight sequences to build a directed graph. The nodes represent significant motifs, and the edges are weighted by how often one motif directly follows another in the flight data.

**Usage:**

```bash
python network_builder.py -k 5 --top_edges 1500 --html_out "html-graphs/my_network.html"

```

**Command-Line Arguments:**
| Argument | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--k` | `-k` | **Required** | **Motif length:** Must perfectly match the `-k` used in Step 1. |
| `--html_out` | | `html-graphs/motif_network.html` | Filepath for the generated interactive HTML graph. |
| `--top_edges`| | `2000` | The maximum number of strongest edges to include in the visualization. |
| `--flights_file`| | `flights.json` | Path to the flights artifact generated in Step 1. |
| `--motifs_file`| | `significant_motifs.csv` | Path to the motifs artifact generated in Step 1. |
| `--network_csv`| | `final_network.csv` | Path to save the intermediate raw edge list. |

**Outputs:**

1. `final_network.csv`: The complete calculated edgelist showing source motifs, target motifs, and their transition weights.
2. An interactive HTML graph saved to your specified `--html_out` location.

---

### Step 3: Flight Clustering (`flight_clusterer.py`)

This script vectorizes the flight sequences using TF-IDF (treating motifs as "words" and flights as "documents") and groups them using KMeans clustering.

**Usage:**

```bash
python flight_clusterer.py -k 5 -c 4 --output_csv "flight_clusters.csv"

```

**Command-Line Arguments:**
| Argument | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--k` | `-k` | `5` | **Motif length:** Must perfectly match the `-k` used in Step 1. |
| `--n_clusters` | `-c` | `4` | **Number of clusters:** How many groups to segment the flights into. |
| `--flight_info`| | `flight_info.csv` | Path to the flight metadata artifact generated in Step 1. |
| `--motifs_file`| | `significant_motifs.csv` | Path to the motifs artifact generated in Step 1. |
| `--flights_file`| | `flights.json` | Path to the flights artifact generated in Step 1. |
| `--output_csv` | | `flight_clusters.csv` | Filepath to save the final clustering results. |

**Outputs:**

1. Prints cluster demographics (aircraft, operator, mission percentages) and top motifs per cluster to the terminal.
2. `flight_clusters.csv`: A dataframe mapping each `flight_id` to its assigned `cluster`.

---

## Input Data Format (For Step 1)

The initial script expects a **semicolon-separated (`;`)** CSV file containing a header row and an index column.

The dataset must contain the following columns and format:

|  | F_SESSION | F_START_FRAME | F_END_FRAME | F_DURATION | FIRST_WORD_INDEX | SECOND_WORD_INDEX | THIRD_WORD_INDEX | session | k_aircraft | k_operator | k_mission |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 3152409 | 742 | 789 | 00:00:24.000 | 3.0 | 10.0 | 285.0 | 3152409 | 41622 | 81 | 1 |
| **1** | 3152409 | 790 | 855 | 00:00:33.000 | 3.0 | 75.0 | 281.0 | 3152409 | 41622 | 81 | 1 |

*Note: The script automatically drops the `session` column and reconstructs the phase sequences using the `FIRST`, `SECOND`, and `THIRD_WORD_INDEX` columns.*

```

---

You now have a robust, production-ready project structure! Would you like me to write a quick `run_pipeline.sh` bash script so you can execute all three steps sequentially with a single terminal command?

```