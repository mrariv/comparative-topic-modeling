from gensim.models import CoherenceModel

def compute_coherence(topics_words, tokens_list, dictionary):
    cm = CoherenceModel(
        topics=topics_words,
        texts=tokens_list,
        dictionary=dictionary,
        coherence="c_v"
    )
    
    return cm.get_coherence(), cm.get_coherence_per_topic()

def compute_diversity(model, top_k=10):
    info = model.get_topic_info()
    topics = info[info["Topic"] != -1]["Topic"].tolist()

    all_words = []
    for t in topics:
        ws = [w for w, _ in (model.get_topic(t) or [])[:top_k]]
        all_words.extend(ws)

    if not all_words:
        return 0.0
    
    return len(set(all_words)) / len(all_words)

def compute_lda_diversity(lda, top_k=10):
    topics = lda.show_topics(num_topics=lda.num_topics, num_words=top_k, formatted=False)
    top_words = [[w for w, _ in words] for _, words in topics]
    all_words = [w for t in top_words for w in t]
    return len(set(all_words)) / len(all_words)