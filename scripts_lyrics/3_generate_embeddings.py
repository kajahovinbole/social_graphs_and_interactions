import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path("data_lyrics/lyrics.csv")
OUTPUT_DIR = Path("data_lyrics/embeddings/")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

BATCH_SIZE = 32

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = SentenceTransformer(MODEL_NAME)

# Account for special tokens added by the tokenizer.
# This ensures chunks do not exceed the model's max sequence length.
MAX_TOKENS = (
    model.max_seq_length
    - model.tokenizer.num_special_tokens_to_add(pair=False)
)

print(f"Model max sequence length: {model.max_seq_length}")
print(f"Maximum content tokens per chunk: {MAX_TOKENS}")


# --------------------------------------------------
# Split lyrics into chunks
# --------------------------------------------------

def split_into_chunks(text, tokenizer, max_tokens):
    tokens = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False
    )["input_ids"]

    chunks = [
        tokens[i:i + max_tokens]
        for i in range(0, len(tokens), max_tokens)
    ]

    return [
        tokenizer.decode(chunk, skip_special_tokens=True)
        for chunk in chunks
    ]


# --------------------------------------------------
# Generate song embedding
# --------------------------------------------------

def get_song_embedding(lyrics, model):
    chunks = split_into_chunks(
        lyrics,
        model.tokenizer,
        MAX_TOKENS
    )

    if not chunks:
        return None

    embeddings = model.encode(
        chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    weights = np.array([
        len(model.tokenizer(
            chunk,
            add_special_tokens=False
        )["input_ids"])
        for chunk in chunks
    ])

    song_embedding = np.average(
        embeddings,
        axis=0,
        weights=weights
    )

    return song_embedding


# --------------------------------------------------
# Generate artist embedding
# --------------------------------------------------

def get_artist_embedding(song_embeddings):
    artist_embedding = np.mean(song_embeddings, axis=0)

    norm = np.linalg.norm(artist_embedding)

    if norm > 0:
        artist_embedding = artist_embedding / norm

    return artist_embedding


# --------------------------------------------------
# Load and prepare lyrics
# --------------------------------------------------

def load_lyrics():
    df = pd.read_csv(INPUT_FILE)

    # Keep only rows with actual lyrics
    df = df[
        (df["found"] == True)
        & df["lyrics_text"].notna()
        & (df["lyrics_text"].str.strip() != "")
        & df["artist_mbid"].notna()
    ].copy()

    # Remove duplicate recordings
    df = df.drop_duplicates(
        subset="recording_mbid",
        keep="first"
    )

    print(f"Songs with valid lyrics: {len(df)}")
    print(f"Unique artists: {df['artist_mbid'].nunique()}")

    return df


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_lyrics()

    song_embeddings = []
    song_metadata = []

    # Generate one embedding per song
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Generating song embeddings"
    ):
        embedding = get_song_embedding(
            row["lyrics_text"],
            model
        )

        if embedding is None:
            continue

        song_embeddings.append(embedding)

        song_metadata.append({
            "recording_mbid": row["recording_mbid"],
            "artist_mbid": row["artist_mbid"],
            "artist": row["artist"],
            "recording_name": row["recording_name"]
        })

    song_embeddings = np.array(song_embeddings)
    song_metadata = pd.DataFrame(song_metadata)

    # Save song embeddings and metadata
    np.save(
        OUTPUT_DIR / "song_embeddings.npy",
        song_embeddings
    )

    song_metadata.to_csv(
        OUTPUT_DIR / "song_metadata.csv",
        index=False
    )

    print(f"\nSong embeddings shape: {song_embeddings.shape}")

    # --------------------------------------------------
    # Aggregate song embeddings into artist embeddings
    # --------------------------------------------------

    artist_embeddings = []
    artist_metadata = []

    for artist_mbid, group in tqdm(
        song_metadata.groupby("artist_mbid"),
        desc="Generating artist embeddings"
    ):
        indices = group.index.to_numpy()

        artist_song_embeddings = song_embeddings[indices]

        artist_embedding = get_artist_embedding(
            artist_song_embeddings
        )

        artist_embeddings.append(artist_embedding)

        artist_metadata.append({
            "artist_mbid": artist_mbid,
            "artist": group["artist"].iloc[0],
            "num_songs": len(group)
        })

    artist_embeddings = np.array(artist_embeddings)
    artist_metadata = pd.DataFrame(artist_metadata)

    # Save artist embeddings and metadata
    np.save(
        OUTPUT_DIR / "artist_embeddings.npy",
        artist_embeddings
    )

    artist_metadata.to_csv(
        OUTPUT_DIR / "artist_metadata.csv",
        index=False
    )

    print("\nFinished!")
    print(f"Artist embeddings shape: {artist_embeddings.shape}")
    print(f"Artists included: {len(artist_metadata)}")

    print(f"\nFiles saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
