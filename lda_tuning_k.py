from src.config import DATA_PATH, EMBEDDINGS_DIR, LEMMATIZATION_DIR, OUTPUTS_DIR
from src.data import load_corpus
from src.embeddings import generate_openai_embeddings, generate_sbert_embeddings
from src.lemmatization import lemmatize_texts
from src.modeling import fit_lda, fit_bertopic
from src.evaluation import compute_coherence, compute_diversity, compute_lda_diversity
import pandas as pd
from gensim.corpora import Dictionary
import numpy as np
import os
import json
from gensim.models import CoherenceModel

if __name__ == '__main__':

    # 1. Loading data and collecting corpus
    df = load_corpus(path=DATA_PATH)

    print("Dataset is loaded")

    texts = df["text"].tolist()
    lemmatized_texts = lemmatize_texts(texts, output_path=f"{LEMMATIZATION_DIR}processed_texts.csv")

    print("Text is lemmatized")

    lemmatized_texts = [t if isinstance(t, str) and t.strip() else "" for t in lemmatized_texts]
    tokens_list = [text.split() for text in lemmatized_texts if text.strip()]
    dictionary = Dictionary(tokens_list)

    print("Dictionary is collected")

    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

    print("Corpus is created")

    # 2. Fitting LDA and storing results
    results_path = "outputs/lda_tuning_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}

    k_values = range(10, 81, 5)

    print("For-loop has started")

    for k in k_values:
        if str(k) in results:
            print(f"K={k} already computed, skipping")
            continue

        print("LDA model fitting started")
        
        lda_model = fit_lda(tokens_list, dictionary, corpus, num_topics=k)

        print("LDA model fitting finished")

        topics_words_lda = [
            [word for word, _ in lda_model.show_topic(topic_id, topn=10)]
            for topic_id in range(lda_model.num_topics)
            ]
        
        print(topics_words_lda)

        # lda_coherence, _ = compute_coherence(topics_words_lda, tokens_list, dictionary)

        cm = CoherenceModel(
            topics=topics_words_lda,
            texts=tokens_list,
            dictionary=dictionary,
            coherence="c_v"
        )

        print("CoherenceModel finished")

        lda_coherence = cm.get_coherence()

        print("LDA coherence:", lda_coherence)

        results[str(k)] = lda_coherence
        print(f"K={k}, coherence={lda_coherence:.4f}")

        with open(results_path, 'w') as f:
            json.dump(results, f)

    best_k = max(results, key=lambda k: results[k])
    print(f"\nBest K: {best_k}, coherence={results[best_k]:.4f}")