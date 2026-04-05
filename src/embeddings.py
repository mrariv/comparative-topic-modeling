from config import EMBEDDINGS_DIR, RANDOM_SEED, BATCH_SIZE, OPENAI_MODEL, SBERT_MODEL
from dotenv import load_dotenv
from openai import OpenAI
import os
from tqdm import tqdm
import numpy as np
import time
from sentence_transformers import SentenceTransformer

load_dotenv()

def generate_openai_embeddings(df, output_path):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("OpenAI connected!")

    texts = df['text_for_llm'].tolist()

    if os.path.exists(output_path):
        embeddings = list(np.load(output_path))
        start = len(embeddings)
        print(f"Resuming from {start} embeddings")
    else:
        embeddings = []
        start = 0

    for i in tqdm(range(start, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]

        try:
            response = client.embeddings.create(
                model=OPENAI_MODEL,
                input=batch
            )
        
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            time.sleep(2)
            continue

        if i % 100 == 0:
            np.save(output_path, np.array(embeddings, dtype=np.float32))
            print(f"Done: {len(embeddings)} OpenAI embeddings saved to {output_path}")

    return True


def generate_sbert_embeddings(df, output_path):
    model = SentenceTransformer(SBERT_MODEL)
    texts = df['text_for_llm'].tolist()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
        )
    
    embeddings = embeddings.astype(np.float32)
    np.save(output_path, embeddings)
    print(f"Done: {len(embeddings)} SBERT embeddings saved to {output_path}")

    return True