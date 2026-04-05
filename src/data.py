from src.config import DATA_PATH, MAX_CHARS
import pandas as pd

def load_corpus(path=DATA_PATH):
    # 1. Load data
    df = pd.read_csv(path)

    # 2. Convert dates to datetime dtype
    df['date'] = pd.to_datetime(df['date'], errors="coerce")

    # 3. Drop url duplicates
    df = df.drop_duplicates(subset=['url'])

    # 4. Drop rows where text is null
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != ""]

    # 5. Create a text_for_llm column
    df['text_for_llm'] = df.apply(
        lambda x: "\n".join([str(x['title'] or ""), str(x['text'] or "")])[:MAX_CHARS],
        axis=1
        )
    
    print(f"Loaded corpus: {df.shape[0]} documents")
    return df