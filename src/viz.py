import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
from src.config import RANDOM_SEED

def plot_topic_sizes(topics, title):
    s = pd.Series(topics)
    s = s[s != -1].value_counts()
    
    plt.figure(figsize=(8, 4))
    s.sort_values(ascending=False).head(15).plot(kind="bar")
    plt.title(title)
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.show()

def plot_umap(embeddings, topics, title):
    umap = UMAP(
        n_neighbors=15,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_SEED
    )

    emb_2d = umap.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=topics,
        cmap="tab20",
        s=5,
        alpha=0.6
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()