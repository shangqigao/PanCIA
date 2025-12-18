import pandas as pd 
import json

alpha = 0.005
signature = ["GeneProgrames", "HRDscore", "ImmuneSignatureScore", "StemnessScoreDNA", "StemScoreRNA"][4]
root_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/results"
if signature == "GeneProgrames":
    json_path = f"{root_dir}/TCGA_gene_program.json"
elif signature == "HRDscore":
    json_path = f"{root_dir}/TCGA_HRD_score.json"
elif signature == "ImmuneSignatureScore":
    json_path = f"{root_dir}/TCGA_immune_signature.json"
elif signature == "StemnessScoreDNA":
    json_path = f"{root_dir}/TCGA_DNA_stemness_score.json"
elif signature == "StemScoreRNA":
    json_path = f"{root_dir}/TCGA_RNA_stemness_score.json"

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
df = df.where(df < alpha)
row_counts = (df < alpha).sum(axis=1)
col_counts = (df < alpha).sum(axis=0)
df['row_count'] = row_counts
df.loc['col_count'] = col_counts
col_count_row = df.loc[['col_count']]
main_rows = df.drop(index='col_count')
main_rows_sorted = main_rows.sort_values(by='row_count', ascending=False)
df_sorted = pd.concat([main_rows_sorted, col_count_row], axis=0)
# Save to Excel
save_path = json_path.replace('.json', '.xlsx')
df_sorted.to_excel(save_path)
print(df_sorted)