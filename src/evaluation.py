from gensim.models import CoherenceModel
import random

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

def save_word_intrusion_stimuli(topics_words, model_name, n_topics=10, topn=5):
    selected = [t for t in topics_words[:15] if len(set(t)) >= 5][:n_topics]
    stimuli = []

    for i, topic_words in enumerate(selected):
        other_topics = [t for j, t in enumerate(selected) if j != i]
        intruder_pool = [w for t in other_topics for w in t[:3]]
        intruder = random.choice([w for w in intruder_pool if w not in topic_words])

        clean_words = list(dict.fromkeys(topic_words[:topn]))[:5]
        words = clean_words + [intruder]
        random.shuffle(words)

        stimuli.append({
            "model": model_name,
            "topic_id": i,
            "words": words,
            "intruder": intruder
        })
    
    return stimuli