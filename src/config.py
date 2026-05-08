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
LDA_N_TOPICS = 30  # tuned with lda_tuning_k.py run on a full corpus

# BERTopic
HDBSCAN_MIN_CLUSTER_SIZE = 500  # tuned with bertopic_tuning_k.py on a full corpus

# OpenAI
OPENAI_MODEL = "text-embedding-3-small"

# BERTopic
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"