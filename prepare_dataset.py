import os
import json
from PIL import Image
from datasets import Dataset
import pandas as pd

# Load caption.json
with open("caption.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare list for HF dataset
records = []
for item in data:
    path = os.path.join("dataset", item["file"])
    if os.path.exists(path):
        records.append({"image": path, "text": item["text"]})

# Save as CSV for HuggingFace Dataset
df = pd.DataFrame(records)
df.to_csv("hf_data.csv", index=False)
print("Hugging Face dataset ready.")
