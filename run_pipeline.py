from src.config import DATA_PATH, EMBEDDINGS_DIR, LEMMATIZATION_DIR, OUTPUTS_DIR, RANDOM_SEED
from src.data import load_corpus
from src.embeddings import generate_openai_embeddings, generate_sbert_embeddings
from src.lemmatization import lemmatize_texts
from src.modeling import fit_lda, fit_bertopic, safe_split
from src.evaluation import compute_coherence, compute_diversity, compute_lda_diversity, save_word_intrusion_stimuli
import pandas as pd
import random
import json
from gensim.corpora import Dictionary

def bertopic_topics_to_words(model, dictionary, topn=10):
    topics_dict = model.get_topics()
    result = []
    for topic_id, words in topics_dict.items():
        if topic_id == -1 or not words:
            continue
        clean_words = []
        for w, _ in words[:topn*2]:
            tokens = w.replace('«', '').replace('»', '').replace('"', '').replace('"', '').replace('.', '').replace(',', '').strip().split()
            for t in tokens:
                if t in dictionary.token2id and len(t) >= 2:
                    clean_words.append(t)
        if clean_words:
            result.append(clean_words[:topn])
    return result

def main():
    random.seed(RANDOM_SEED)
    # 1. Loading corpus
    df = load_corpus(path=DATA_PATH)

    # 2. Lemmatizing texts (for LDA)
    texts = df["text"].tolist()
    lemmatized_texts = lemmatize_texts(texts, output_path=f"{LEMMATIZATION_DIR}processed_texts.csv")

    # 3. Generating embeddings
    openai_embeddings = generate_openai_embeddings(df, f"{EMBEDDINGS_DIR}openai_embeddings.npy")
    sbert_embeddings = generate_sbert_embeddings(df, f"{EMBEDDINGS_DIR}sbert_embeddings.npy")

    # 4. Fitting models
    lemmatized_texts = [t if isinstance(t, str) and t.strip() else "" for t in lemmatized_texts]
    tokens_list = [text.split() for text in lemmatized_texts if text.strip()]
    dictionary = Dictionary(tokens_list)
    
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    lda_model = fit_lda(tokens_list, dictionary, corpus)

    docs_original = pd.Series(df['text_for_llm']).fillna("").astype(str)

    original_tokens_list = [safe_split(text) for text in docs_original if text.strip()]
    original_dictionary = Dictionary(original_tokens_list)
    original_dictionary.filter_extremes(no_below=10, no_above=0.5)
    
    sbert_model, sbert_topics, sbert_probs = fit_bertopic(docs_original, sbert_embeddings)
    openai_model, openai_topics, openai_probs = fit_bertopic(docs_original, openai_embeddings)

    # 5. Evaluation
    topics_words_lda = [
        [word for word, _ in lda_model.show_topic(topic_id, topn=10)]
        for topic_id in range(lda_model.num_topics)
        ]

    lda_coherence, _ = compute_coherence(topics_words_lda, tokens_list, dictionary)
    lda_diversity = compute_lda_diversity(lda_model, top_k=10)

    topics_words_sbert = bertopic_topics_to_words(sbert_model, original_dictionary, topn=10)
    sbert_coherence, _ = compute_coherence(topics_words_sbert, original_tokens_list, original_dictionary)
    sbert_diversity = compute_diversity(sbert_model, top_k=10)

    topics_words_openai = bertopic_topics_to_words(openai_model, original_dictionary, topn=10)
    openai_coherence, _ = compute_coherence(topics_words_openai, original_tokens_list, original_dictionary)
    openai_diversity = compute_diversity(openai_model, top_k=10)

    # 6.1 Store model results
    lda_model.save(f"{OUTPUTS_DIR}lda_model")
    sbert_model.save(f"{OUTPUTS_DIR}bertopic_sbert")
    openai_model.save(f"{OUTPUTS_DIR}bertopic_openai")

    # save topic words
    sbert_model.get_topic_info().to_csv(f"{OUTPUTS_DIR}topics_sbert.csv", index=False)
    openai_model.get_topic_info().to_csv(f"{OUTPUTS_DIR}topics_openai.csv", index=False)

    # save LDA topics
    lda_topics = [(i, lda_model.show_topic(i, topn=20)) for i in range(lda_model.num_topics)]
    pd.DataFrame([{"topic_id": t, "words": str(w)} for t, w in lda_topics]).to_csv(f"{OUTPUTS_DIR}topics_lda.csv", index=False)

    # 6.2 Store evaluation results
    print("\n=== RESULTS ===")
    print(f"LDA — coherence: {lda_coherence:.4f}, diversity: {lda_diversity:.4f}")
    print(f"SBERT — coherence: {sbert_coherence:.4f}, diversity: {sbert_diversity:.4f}")
    print(f"OpenAI — coherence: {openai_coherence:.4f}, diversity: {openai_diversity:.4f}")

    stimuli = []
    stimuli += save_word_intrusion_stimuli(topics_words_lda, "LDA", topn=7)
    stimuli += save_word_intrusion_stimuli(topics_words_sbert, "SBERT", topn=7)
    stimuli += save_word_intrusion_stimuli(topics_words_openai, "OpenAI", topn=7)

    random.shuffle(stimuli)

    with open(f"{OUTPUTS_DIR}word_intrusion_stimuli.json", "w", encoding="utf-8") as f:
        json.dump(stimuli, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(stimuli)} stimuli for word intrusion task")

    results = pd.DataFrame({
        "model": ["LDA", "SBERT", "OpenAI"],
        "coherence": [lda_coherence, sbert_coherence, openai_coherence],
        "diversity": [lda_diversity, sbert_diversity, openai_diversity]
    })
    results.to_csv(f"{OUTPUTS_DIR}results.csv", index=False)
    print(f"Results saved to {OUTPUTS_DIR}results.csv")

if __name__ == "__main__":
    main()