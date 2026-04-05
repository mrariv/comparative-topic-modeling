# src/config.py
DATA_PATH = "data/lenta_socioeconomic_corpus.csv"
EMBEDDINGS_DIR = "embeddings/"
LEMMATIZATION_DIR = "lemmatized/"
OUTPUTS_DIR = "outputs/"

# Model settings
RANDOM_SEED = 42
MAX_CHARS = 4000
BATCH_SIZE = 32

# LDA
LDA_N_TOPICS = 45  # to be tuned on full corpus

# BERTopic
HDBSCAN_MIN_CLUSTER_SIZE = 30  # this controls topic granularity, not n_topics directly

# OpenAI
OPENAI_MODEL = "text-embedding-3-small"

# BERTopic
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"