from src.config import DATA_PATH, EMBEDDINGS_DIR, LEMMATIZATION_DIR, OUTPUTS_DIR
from src.data import load_corpus
from src.embeddings import generate_openai_embeddings, generate_sbert_embeddings
from src.lemmatization import lemmatize_texts
from src.modeling import fit_lda, fit_bertopic
from src.evaluation import compute_coherence, compute_diversity, compute_lda_diversity
import pandas as pd
from gensim.corpora import Dictionary

def bertopic_topics_to_words(model, topn=10):
    topics_dict = model.get_topics()

    return [
        [w for w, _ in words[:topn]]
        for topic_id, words in topics_dict.items()
        if topic_id != -1
    ]

def main():
    # 1. Loading corpus
    df = load_corpus(path=DATA_PATH)

    # 2. Lemmatizing texts (for LDA)
    texts = df["text"].tolist()
    lemmatized_texts = lemmatize_texts(texts, output_path=f"{LEMMATIZATION_DIR}processed_texts")

    # 3. Generating embeddings
    openai_embeddings = generate_openai_embeddings(df, f"{EMBEDDINGS_DIR}openai_embeddings")
    sbert_embeddings = generate_sbert_embeddings(df, f"{EMBEDDINGS_DIR}sbert_embeddings")

    # 4. Fitting models
    tokens_list = [text.split() for text in lemmatized_texts]
    dictionary = Dictionary(tokens_list)
    
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    lda_model = fit_lda(tokens_list, dictionary, corpus)

    docs = df['text_for_llm'].fillna("").astype(str)
    sbert_model, sbert_topics, sbert_probs = fit_bertopic(docs, sbert_embeddings)
    openai_model, openai_topics, openai_probs = fit_bertopic(docs, openai_embeddings)

    # 5. Evaluation
    topics_words_lda = [
        [word for word, _ in lda_model.show_topic(topic_id, topn=10)]
        for topic_id in range(lda_model.num_topics)
        ]

    lda_coherence, _ = compute_coherence(topics_words_lda, tokens_list, dictionary)
    lda_diversity = compute_lda_diversity(lda_model, top_k=10)

    topics_words_sbert = bertopic_topics_to_words(sbert_model, topn=10)
    sbert_coherence, _ = compute_coherence(topics_words_sbert, tokens_list, dictionary)
    sbert_diversity = compute_diversity(sbert_model, top_k=10)

    topics_words_openai = bertopic_topics_to_words(openai_model, topn=10)
    openai_coherence, _ = compute_coherence(topics_words_openai, tokens_list, dictionary)
    openai_diversity = compute_diversity(openai_model, top_k=10)

    print("\n=== RESULTS ===")
    print(f"LDA — coherence: {lda_coherence:.4f}, diversity: {lda_diversity:.4f}")
    print(f"SBERT — coherence: {sbert_coherence:.4f}, diversity: {sbert_diversity:.4f}")
    print(f"OpenAI — coherence: {openai_coherence:.4f}, diversity: {openai_diversity:.4f}")

    results = pd.DataFrame({
        "model": ["LDA", "SBERT", "OpenAI"],
        "coherence": [lda_coherence, sbert_coherence, openai_coherence],
        "diversity": [lda_diversity, sbert_diversity, openai_diversity]
    })
    results.to_csv(f"{OUTPUTS_DIR}results.csv", index=False)
    print(f"Results saved to {OUTPUTS_DIR}results.csv")

if __name__ == "__main__":
    main()