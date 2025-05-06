import os
import json
import pandas as pd

with open("caption.json", "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for item in data:
    image_path = os.path.join("dataset", item["file"])
    if os.path.exists(image_path):
        records.append({"image": image_path, "text": item["text"]})
    else:
        print("⚠️ Missing image:", image_path)

df = pd.DataFrame(records)
df.to_csv("train_data.csv", index=False)
print("✅ train_data.csv generated.")
