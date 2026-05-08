# Comparative Topic Modeling of Russian Socio-Economic News

A research pipeline for comparing three topic modeling paradigms on a large Russian-language news corpus:
- **LDA** (Latent Dirichlet Allocation) via Gensim
- **BERTopic** with multilingual sentence-transformer embeddings (SBERT)
- **BERTopic** with OpenAI LLM-based embeddings

## Research Context

This pipeline supports a bachelor's thesis at HSE University comparing probabilistic, transformer-based, and LLM-based topic modeling approaches on 144,475 Russian-language socio-economic news articles from Lenta.ru (1999–2024).

## Project Structure

```
comparative-topic-modeling/
├── data/                          # Corpus data (gitignored)
├── embeddings/                    # Generated embeddings (gitignored)
├── lemmatized/                    # Lemmatized texts (gitignored)
├── outputs/
│   ├── lda_model*                 # Saved Gensim LDA model
│   ├── bertopic_sbert/            # Saved BERTopic (SBERT) model
│   ├── bertopic_openai/           # Saved BERTopic (OpenAI) model
│   ├── topics_lda.csv             # LDA topic word lists
│   ├── topics_sbert.csv           # BERTopic (SBERT) topic info
│   ├── topics_openai.csv          # BERTopic (OpenAI) topic info
│   ├── lda_tuning_results.json    # LDA k-tuning results
│   ├── bertopic_tuning_results*.json  # BERTopic min_cluster_size tuning
│   ├── word_intrusion_stimuli.json    # Main word intrusion stimuli
│   ├── word_intrusion_extended/   # Extended questionnaires (3 files)
│   └── results.csv                # Final evaluation metrics
├── src/
│   ├── config.py                  # Paths and hyperparameters
│   ├── data.py                    # Corpus loading and preprocessing
│   ├── lemmatization.py           # Russian text lemmatization (pymorphy2)
│   ├── embeddings.py              # OpenAI and SBERT embedding generation
│   ├── modeling.py                # LDA and BERTopic model fitting
│   ├── evaluation.py              # Coherence and diversity metrics
│   └── viz.py                     # Topic size and UMAP visualizations
├── notebooks/
│   ├── data_collection_lentaru.ipynb   # Corpus assembly from two sources
│   ├── centroid_analysis.ipynb         # Topic centroid exploration
│   └── visualizations.ipynb            # Results visualizations
├── sbert_embeddings_generation.ipynb   # SBERT embedding generation (Colab)
├── run_pipeline.py                # Main pipeline entry point
├── recompute_metrics.py           # Recompute metrics from saved models
├── lda_tuning_k.py                # Grid search over LDA topic count k
├── bertopic_tuning_k.py           # Grid search over HDBSCAN min_cluster_size
├── generate_word_intrusion.py     # Regenerate main word intrusion stimuli
├── generate_word_intrusion_extended.py  # Generate extended questionnaires
├── test_pipeline.py               # End-to-end test on 100-document subset
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/mrariv/comparative-topic-modeling.git
cd comparative-topic-modeling

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

echo "OPENAI_API_KEY=your_key_here" > .env
```

## Data

The corpus consists of 144,475 Russian-language news articles tagged as **Экономика** or **Бизнес** from the Lenta.ru archive, covering 1999–2024. The corpus is not included in the repository.

Sources:
- [Lenta.ru corpus 1999–2019](https://github.com/yutkin/Lenta.Ru-News-Dataset)
- [Lenta.ru corpus 2019–2024](https://huggingface.co/datasets/data-silence/lenta.ru_2-extended)

Place the combined CSV at `data/lenta_socioeconomic_corpus.csv`.

## Running the Pipeline

**Test on 100-document subset first:**
```bash
python3 -W ignore test_pipeline.py
```

**Run full pipeline:**
```bash
python3 -W ignore run_pipeline.py
```

Results are saved to `outputs/results.csv`.

## Pipeline Steps

1. **Load corpus** — filters, deduplicates, creates `text_for_llm` field
2. **Lemmatize** — cleans text, tokenizes with `razdel`, removes stopwords, lemmatizes with `pymorphy2`. Checkpoints every 1000 documents.
3. **Generate embeddings** — OpenAI `text-embedding-3-small` (with resumable checkpointing) and `paraphrase-multilingual-MiniLM-L12-v2` via sentence-transformers
4. **Fit models** — LDA (30 topics) on lemmatized tokens; BERTopic on original `text_for_llm` text with pre-computed embeddings
5. **Evaluate** — topic coherence (c_v) and topic diversity (top-10 words) for all three models
6. **Save results** — models to `outputs/`, metrics to `outputs/results.csv`, word intrusion stimuli to `outputs/word_intrusion_stimuli.json`

## Hyperparameter Tuning

Both LDA and BERTopic hyperparameters were tuned on the full corpus before the final run:

- **`lda_tuning_k.py`** — sweeps `k` from 10 to 80 in steps of 5, picking the value with highest c_v coherence. Results cached in `outputs/lda_tuning_results.json`.
- **`bertopic_tuning_k.py`** — sweeps `min_cluster_size` over `[10, 20, 50, 100, …, 1000]`, targeting ~30 topics while minimising outlier rate. Results cached in `outputs/bertopic_tuning_results*.json`.

## BERTopic Architecture

BERTopic is configured with:
- **UMAP** — `n_neighbors=15`, `n_components=5`, `metric=cosine`
- **HDBSCAN** — `min_cluster_size=500`, `min_samples=max(5, min_cluster_size//3)`, `cluster_selection_method=eom`
- **CountVectorizer** — unigrams + bigrams (`ngram_range=(1,2)`), `min_df=5`, `max_df=0.7`
- **c-TF-IDF** — BM25 weighting + frequent-word reduction (`ClassTfidfTransformer`)

## Evaluation

Two coherence scores are computed for cross-model comparison:

| Score | Reference corpus | Purpose |
|-------|-----------------|---------|
| **Native coherence** | Each model's own vocabulary | Measures within-model topic quality |
| **Comparable coherence** | Shared lemmatized corpus | Fair cross-model comparison |

Use `recompute_metrics.py` to recompute both scores from already-saved models without re-running the full pipeline.

## Word Intrusion Task

Word intrusion is used as a human evaluation proxy for topic interpretability.

- **`generate_word_intrusion.py`** — regenerates the main stimuli file from saved models. Topics are coherence-filtered (bottom 40th percentile dropped), intruders are drawn from the most Jaccard-distant topic and frequency-matched to avoid detection cues.
- **`generate_word_intrusion_extended.py`** — generates three balanced questionnaire files covering all remaining topics (no coherence filter), enabling correlation analysis between coherence scores and human word intrusion accuracy.

Stimuli are saved to `outputs/word_intrusion_stimuli.json` and `outputs/word_intrusion_extended/`.

## Key Configuration

All hyperparameters are in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LDA_N_TOPICS` | 30 | Number of LDA topics (tuned) |
| `HDBSCAN_MIN_CLUSTER_SIZE` | 500 | BERTopic cluster granularity (tuned) |
| `BATCH_SIZE` | 32 | Embedding batch size |
| `MAX_CHARS` | 4000 | Max characters per document |
| `OPENAI_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `SBERT_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | Sentence transformer model |

## Notes

- SBERT embedding generation requires GPU for reasonable speed. Use `sbert_embeddings_generation.ipynb` on Google Colab and save the `.npy` file to `embeddings/`.
- OpenAI embedding generation costs approximately $1.50 for the full corpus and supports resuming from checkpoints.
- Lemmatization takes approximately 1–2 hours on CPU for the full corpus.
