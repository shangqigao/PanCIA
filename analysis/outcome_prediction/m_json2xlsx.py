import pandas as pd
import json
import pathlib

json_path = "1st-contrast.json"

# Load JSON
with open(json_path) as f:
    data = json.load(f)

# Flatten nested structure into a dataframe
df = pd.json_normalize(
    data,
    sep="."  # flatten nested keys with dots
)

# Transpose to get nested keys as rows
df = df.T.reset_index()
df.columns = ["path", "value"]

# Split the path into index + metrics
df["metric"] = df["path"].str.split(".").str[-1]
df["index"] = df["path"].str.replace(r"\.[^.]+$", "", regex=True)

# Pivot to make metrics into columns
df = df.pivot(index="index", columns="metric", values="value")

# Save to Excel
name = pathlib.Path(json_path).stem
df.to_excel(f"{name}.xlsx")
print(df)
