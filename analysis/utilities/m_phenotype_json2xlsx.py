import pandas as pd
import json
import pathlib
import numpy as np

phenotype = ["ImmuneSubtype", "MolecularSubtype", "PrimaryDisease"][2]
root_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/PanCIA/figures/results/"
if phenotype == 'ImmuneSubtype':
    json_path = f"{root_dir}/TCGA_immune_subtype.json"
    indices = {0: 'IFN-gamma Dominant (Immune C2)', 1: 'Inflammatory (Immune C3)', 2: 'Wound Healing (Immune C1)', 3: 'Lymphocyte Depleted (Immune C4)'}
elif phenotype == 'MolecularSubtype':
    json_path = f"{root_dir}/TCGA_molecular_subtype.json"
    indices = {0: 'BRCA.LumA', 1: 'BRCA.LumB', 2: 'BRCA.Basal', 3: 'BRCA.Normal', 4: 'GI.CIN', 5: 'KIRC.1', 6: 'KIRC.2', 7: 'KIRC.3', 8: 'KIRC.4', 9: 'LIHC.iCluster:3', 10: 'LIHC.iCluster:1', 11: 'LIHC.iCluster:2', 12: 'OVCA.Proliferative', 13: 'OVCA.Differentiated', 14: 'OVCA.Mesenchymal', 15: 'UCEC.CN_HIGH'}
elif phenotype == 'PrimaryDisease':
    json_path = f"{root_dir}/TCGA_primary_disease.json"
    indices = {0: 'breast invasive carcinoma', 1: 'bladder urothelial carcinoma', 2: 'ovarian serous cystadenocarcinoma', 3: 'lung adenocarcinoma', 4: 'stomach adenocarcinoma', 5: 'lung squamous cell carcinoma', 6: 'liver hepatocellular carcinoma', 7: 'kidney clear cell carcinoma', 8: 'uterine corpus endometrioid carcinoma', 9: 'cervical & endocervical cancer'}


# Load JSON
with open(json_path) as f:
    data = json.load(f)

def flatten_dict(d, parent_key=""):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


flat = flatten_dict(data)
rows = {}
for full_key, value in flat.items():
    assert isinstance(value, list) and len(value) == len(indices)
    for i, v in enumerate(value):
        index = indices[i]
        if index not in rows:
            rows[index] = {}
        rows[index][full_key] = v

df = pd.DataFrame.from_dict(rows, orient="index")
# Save to Excel
save_path = json_path.replace('.json', '.xlsx')
df.to_excel(save_path)
print(df)