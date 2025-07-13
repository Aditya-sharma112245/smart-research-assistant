from sentence_transformers import SentenceTransformer
import os

model_name = "all-MiniLM-L6-v2"
save_path = "models/all-MiniLM-L6-v2"

os.makedirs(save_path, exist_ok=True)

print("📥 Downloading SentenceTransformer embedding model...")
model = SentenceTransformer(model_name)
model.save(save_path)

print("✅ Embedding model saved to:", save_path)
