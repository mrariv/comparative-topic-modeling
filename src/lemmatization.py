import re
from razdel import tokenize
import pymorphy2
from stop_words import get_stop_words
import pandas as pd
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() # lowercase
    text = re.sub(r"http\S+", "", text) # remove URLs
    text = re.sub(r"[^а-яё\s]", " ", text) # keep only Cyrillic letters
    text = re.sub(r"\s+", " ", text).strip() # normalize spaces
    return text

def tokenize_text(text):
    return [token.text for token in tokenize(text)]

def remove_stopwords(tokens):
    russian_stopwords = set(get_stop_words("russian"))
    return [t for t in tokens if t not in russian_stopwords]

def lemmatize_texts(texts, output_path=None, checkpoint_every=1000):
    if not hasattr(texts, 'apply'):
        texts = pd.Series(texts)

    if output_path and os.path.exists(output_path):
        df_checkpoint = pd.read_csv(output_path)
        results = df_checkpoint["processed_text"].tolist()
        start = len(results)
        print(f"Resuming from {start} documents")
    else:
        results = []
        start = 0

    remaining_texts = texts.iloc[start:]

    if len(remaining_texts) == 0:
        print(f"Already complete: {len(results)} documents")
        return results
    
    remaining_texts = remaining_texts.apply(clean_text)
    remaining_texts = remaining_texts.apply(tokenize_text)
    remaining_texts = remaining_texts.apply(remove_stopwords)
    remaining_texts = remaining_texts.apply(lambda x: " ".join(x))

    morph = pymorphy2.MorphAnalyzer()

    for i, text in enumerate(remaining_texts):
        tokens = text.split()
        lemmas = [morph.parse(token)[0].normal_form for token in tokens]
        results.append(" ".join(lemmas))

        if output_path and (i + 1) % checkpoint_every == 0:
            pd.DataFrame({"processed_text": results}).to_csv(output_path, index=False)
            print(f"Checkpoint: {start + i + 1} documents processed")
    
    if output_path:
        pd.DataFrame({"processed_text": results}).to_csv(output_path, index=False)
        print(f"Completed: {len(results)} documents stored in {output_path}")

    return results