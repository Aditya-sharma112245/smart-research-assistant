from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os

# Define the model name and local save path
model_name = "deepset/bert-base-cased-squad2"
save_path = "models/bert-squad2"  # You can rename this folder if you want

# Make sure the directory exists
os.makedirs(save_path, exist_ok=True)

print("📥 Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

print("📥 Downloading model...")
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.save_pretrained(save_path)

print("✅ Model downloaded and saved to:", save_path)
