from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from src.config import LDA_N_TOPICS, HDBSCAN_MIN_CLUSTER_SIZE, RANDOM_SEED

def safe_split(x):
    x = "" if x is None else str(x)
    return [t for t in x.split() if len(t) >= 2]

def fit_lda(texts, dictionary, corpus):
    num_topics = LDA_N_TOPICS
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=RANDOM_SEED,
        passes=10,
        iterations=200,
        alpha="auto",
        eta="auto",
        chunksize=2000,
        eval_every=None
    )

    return model

def fit_bertopic(docs, embeddings, min_df=5, max_df=0.7, min_cluster_size=None):
    if min_cluster_size is None:
        min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE
    
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        random_state=RANDOM_SEED
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(5, min_cluster_size // 3),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        tokenizer=safe_split,
        preprocessor=None,
        token_pattern=None,
        lowercase=True,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2)
    )

    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True,
        reduce_frequent_words=True
        )

    model = BERTopic(
        language='russian',
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=False,
        verbose=True
        )
    
    topics, probs = model.fit_transform(docs.tolist(), embeddings=embeddings)
    return model, topics, probs