import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, "data")

train_path = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")

print("Loading UNSW train/test CSVs...")
train_df = pd.read_csv(train_path, encoding="latin-1")
test_df = pd.read_csv(test_path, encoding="latin-1")

print("Train:", train_df.shape)
print("Test:", test_df.shape)

# Combine them
df = pd.concat([train_df, test_df], ignore_index=True)

# Standardize the label column name to "Attack Type"
if "label" in df.columns:
    df.rename(columns={"label": "Attack Type"}, inplace=True)
elif "Label" in df.columns:
    df.rename(columns={"Label": "Attack Type"}, inplace=True)

# Strip spaces
df.columns = [c.strip() for c in df.columns]
df["Attack Type"] = df["Attack Type"].astype(str).str.strip()

# Save
out_path = os.path.join(data_dir, "UNSW-NB15.csv")
df.to_csv(out_path, index=False)
print("Saved:", out_path)
