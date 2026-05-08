import numpy as np
from src.data import load_corpus
from src.modeling import fit_bertopic
from src.config import DATA_PATH
import json
import os

if __name__ == '__main__':
    df = load_corpus(path=DATA_PATH)
    docs = df["text"]

    sbert_emb = np.load("embeddings/sbert_embeddings.npy")
    results_path = "outputs/bertopic_tuning_results_min_samples_1.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    
    else:
        results = {}

    mcs_values = [10, 20, 50, 100, 150, 200, 300, 400, 500, 700, 1000]

    for mcs in mcs_values:
        if str(mcs) in results:
            print(f"mcs={mcs} already completed, skipping")
            continue

        model, topics, _ = fit_bertopic(docs, sbert_emb, min_cluster_size=mcs)
        n_topics = len(set(topics)) - 1 # deleting outlier topic
        outliers = sum(1 for t in topics if t == -1)

        print(f"Outliers: {outliers} ({outliers/len(topics) * 100:.1f}%)")
        print(f"min_cluster_size={mcs}, n_topics={n_topics}")

        results[str(mcs)] = {"n_topics": n_topics, "outliers": outliers/len(topics)}

        with open(results_path, "w") as f:
            json.dump(results, f)
    
    best_mcs = min(results, key=lambda k: abs(results[k]["n_topics"] - 30))
    print(f"\nBest mcs={best_mcs}, n_topics={results[best_mcs]['n_topics']}, outliers={results[best_mcs]['outliers']:.1f}")