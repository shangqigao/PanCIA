from pathlib import Path
from collections import Counter
import pandas as pd

root = Path("/data/hdd1/PanCancer/ODELIA")

results = []

for site in sorted(root.iterdir()):

    annotation_path = site / "metadata" / "annotation.csv"
    data_dir = site / "data"

    if not annotation_path.exists():
        continue

    df = pd.read_csv(annotation_path)

    # Select patients with no lesion in either breast
    # across all available examinations
    healthy_ids = (
        df.groupby("PatientID")
        .filter(
            lambda x: (
                (x["Lesion_Left"] == 0) &
                (x["Lesion_Right"] == 0)
            ).all()
        )["PatientID"]
        .unique()
    )

    healthy_df = df[df["PatientID"].isin(healthy_ids)]

    counts = Counter()
    missing = 0

    for uid in healthy_df["UID"].unique():

        patient_dir = data_dir / str(uid)

        if not patient_dir.exists():
            print(f"Missing directory: {patient_dir}")
            missing += 1
            continue

        for scan in patient_dir.rglob("*.nii.gz"):

            # Exclude subtraction images
            if scan.name.startswith("Sub_"):
                continue

            counts["Total scans"] += 1

            if scan.name.startswith("Pre"):
                counts["Pre T1w"] += 1

            elif scan.name.startswith("Post"):
                counts["Post T1w"] += 1

            elif scan.name.startswith("T2"):
                counts["T2w"] += 1

            else:
                counts["Other"] += 1

    results.append({
        "Institution": site.name,
        "Total patients": df["PatientID"].nunique(),
        "No-lesion patients": len(healthy_ids),
        "No-lesion exams": healthy_df["UID"].nunique(),
        "Pre T1w": counts["Pre T1w"],
        "Post T1w": counts["Post T1w"],
        "T2w": counts["T2w"],
        "Other": counts["Other"],
        "Total scans": counts["Total scans"],
        "Missing exams": missing,
    })

# Create summary table
summary = pd.DataFrame(results)

# Add total row
total = summary.select_dtypes(include="number").sum().to_dict()
total["Institution"] = "TOTAL"

summary = pd.concat(
    [summary, pd.DataFrame([total])],
    ignore_index=True
)

print(summary.to_string(index=False))

# Save results
summary.to_csv(
    root / "no_lesion_summary.csv",
    index=False
)