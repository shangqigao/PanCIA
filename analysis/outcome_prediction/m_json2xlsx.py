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

# Reset index if it's not a column
df = df.reset_index()

# Extract fold number
df["fold"] = df["index"].str.extract(r"Fold (\d+)").astype(int)

# Extract ML method
df["method"] = df["index"].str.extract(r"\.(RF|XG|LR|SVC|RSF|CoxPH|Coxnet|FastSVM)\.")  # add your methods

# Extract extractor ID (everything before method)
df["extractor"] = df["index"].str.replace(r"\.(RF|XG|LR|SVC|RSF|CoxPH|Coxnet|FastSVM)\.Fold \d+", "", regex=True)

# List of metric columns
metrics = df.columns.difference(["index", "fold", "method", "extractor"])

# Group by extractor and fold, then take mean of all metric columns
df_mean = df.groupby(["extractor", "fold"], as_index=False)[metrics].mean()

# Save to Excel
name = pathlib.Path(json_path).stem
df_mean.to_excel(f"{name}.xlsx")
print(df)
