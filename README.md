# Comparative Topic Modeling of Russian Socio-Economic News

A research pipeline for comparing three topic modeling paradigms on a large Russian-language news corpus:
- **LDA** (Latent Dirichlet Allocation) via Gensim
- **BERTopic** with multilingual sentence-transformer embeddings
- **BERTopic** with OpenAI LLM-based embeddings

## Research Context

This pipeline supports a bachelor's thesis at HSE University comparing probabilistic, transformer-based, and LLM-based topic modeling approaches on 144,475 Russian-language socio-economic news articles from Lenta.ru (1999–2024).

## Project Structure

```
comparative-topic-modeling/
├── data/                   # Corpus data (gitignored)
├── embeddings/             # Generated embeddings (gitignored)
├── lemmatized/             # Lemmatized texts (gitignored)
├── outputs/                # Results and evaluation metrics
├── src/
│   ├── config.py           # Paths and hyperparameters
│   ├── data.py             # Corpus loading and preprocessing
│   ├── lemmatization.py    # Russian text lemmatization (pymorphy2)
│   ├── embeddings.py       # OpenAI and SBERT embedding generation
│   ├── modeling.py         # LDA and BERTopic model fitting
│   ├── evaluation.py       # Coherence and diversity metrics
│   └── viz.py              # Topic size and UMAP visualizations
├── run_pipeline.py         # Main pipeline entry point
├── test_pipeline.py        # End-to-end test on 100-document subset
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/mrariv/comparative-topic-modeling.git
cd comparative-topic-modeling

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your OpenAI API key
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
4. **Fit models** — LDA (45 topics), BERTopic with SBERT embeddings, BERTopic with OpenAI embeddings
5. **Evaluate** — topic coherence (c_v) and topic diversity (top-10 words) for all three models
6. **Save results** — metrics table to `outputs/results.csv`

## Key Configuration

All hyperparameters are in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LDA_N_TOPICS` | 45 | Number of LDA topics |
| `HDBSCAN_MIN_CLUSTER_SIZE` | 30 | BERTopic cluster granularity |
| `BATCH_SIZE` | 32 | Embedding batch size |
| `MAX_CHARS` | 4000 | Max characters per document |
| `OPENAI_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `SBERT_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | Sentence transformer model |

## Notes

- SBERT embedding generation requires GPU for reasonable speed. Use Google Colab and save the `.npy` file to `embeddings/`.
- OpenAI embedding generation costs approximately $1.50 for the full corpus and supports resuming from checkpoints.
- Lemmatization takes approximately 1–2 hours on CPU for the full corpus.
