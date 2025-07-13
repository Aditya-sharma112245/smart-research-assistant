from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

model_name = "t5-small"
save_path = "models/t5-small"

os.makedirs(save_path, exist_ok=True)

print("📥 Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

print("📥 Downloading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.save_pretrained(save_path)

print("✅ t5-small saved locally at:", save_path)
