import pandas as pd 
import json

json_path = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/results/TCGA_signature.json"
# Load JSON
with open(json_path) as f:
    data = json.load(f)

def flatten_dict(d, parent_key=""):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}+{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


flat = flatten_dict(data)
rows = {}
for full_key, value in flat.items():
    path, col = full_key.rsplit("+", 1)
    if path not in rows:
        rows[path] = {}
    rows[path][col] = value

df = pd.DataFrame.from_dict(rows, orient="index").T
# Save to Excel
save_path = json_path.replace('.json', '.xlsx')
df.to_excel(save_path)
print(df)