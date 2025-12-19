import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# ====================================================
# CONFIGURATION
# ====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-large-patch14"

# ====================================================
# HELPER FUNCTIONS
# ====================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ====================================================
# TEXT EMBEDDING EXTRACTION
# ====================================================
def extract_text_features(df):
    print("\n[INFO] Extracting text embeddings...")

    # Clean and prepare combined text
    df["text_all"] = df["catalog_content"].fillna("").apply(clean_text)
    texts = df["text_all"].tolist()

    # --- TF-IDF ---
    print("[INFO] TF-IDF feature extraction...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(texts)

    svd = TruncatedSVD(n_components=256, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    # --- SBERT ---
    print("[INFO] SentenceTransformer embeddings...")
    sbert = SentenceTransformer(TEXT_MODEL, device=DEVICE)
    sbert_embeddings = sbert.encode(texts, show_progress_bar=True, batch_size=64)

    # --- Numeric Metadata ---
    df["char_count"] = df["text_all"].apply(len)
    df["word_count"] = df["text_all"].apply(lambda x: len(x.split()))
    numeric_feats = df[["char_count", "word_count"]].values

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_feats)

    # --- Combine All ---
    text_features = np.hstack([tfidf_reduced, sbert_embeddings, numeric_scaled])
    print(f"[INFO] Final text feature shape: {text_features.shape}")
    return text_features


# ====================================================
# IMAGE EMBEDDING EXTRACTION
# ====================================================
def extract_image_features(df, image_folder):
    print("\n[INFO] Extracting image embeddings...")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    all_embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = f"{row['sample_id']}.jpg"
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            all_embeddings.append(np.zeros(768))
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                all_embeddings.append(emb.cpu().numpy().flatten())
        except:
            all_embeddings.append(np.zeros(768))
    all_embeddings = np.array(all_embeddings)
    print(f"[INFO] Final image feature shape: {all_embeddings.shape}")
    return all_embeddings


# ====================================================
# MAIN PIPELINE
# ====================================================
def main(data_folder, image_folder):
    csv_path = os.path.join(data_folder, "train.csv") if "train" in data_folder.lower() else os.path.join(data_folder, "test.csv")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")

    os.makedirs("embeddings", exist_ok=True)
    prefix = "train" if "train" in data_folder.lower() else "test"

    # TEXT FEATURES
    text_embeddings = extract_text_features(df)
    np.save(f"embeddings/{prefix}_text.npy", text_embeddings)

    # IMAGE FEATURES
    image_embeddings = extract_image_features(df, image_folder)
    np.save(f"embeddings/{prefix}_image.npy", image_embeddings)

    print(f"\n✅ Done! Embeddings saved under /embeddings as {prefix}_text.npy and {prefix}_image.npy")


# ====================================================
# ENTRY POINT
# ====================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simplified multimodal feature extractor")
    parser.add_argument("--data_folder", required=True, help="Path to train or test folder")
    parser.add_argument("--image_folder", required=True, help="Path to image folder (train_images/test_images)")
    args = parser.parse_args()

    main(args.data_folder, args.image_folder)
