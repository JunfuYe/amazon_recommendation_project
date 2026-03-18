

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

path_2018 = "/content/drive/MyDrive/assignment/table1_interactions_2018.csv"
path_2023 = "/content/drive/MyDrive/assignment/table1_interactions_2023.csv"
output_path = "/content/drive/MyDrive/assignment/table1_interactions.csv"

df18 = pd.read_csv(path_2018)
df23 = pd.read_csv(path_2023)

df18 = df18.rename(columns={
    "reviewerID": "user_id",
    "unixReviewTime": "timestamp",
    "asin": "parent_asin"
})


df18 = df18[["user_id", "parent_asin", "timestamp"]].copy()
df23 = df23[["user_id", "parent_asin", "timestamp"]].copy()

for df in [df18, df23]:
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["parent_asin"] = df["parent_asin"].astype(str).str.strip()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df.dropna(subset=["user_id", "parent_asin", "timestamp"], inplace=True)
    df["timestamp"] = df["timestamp"].astype("int64")

    df.loc[df["timestamp"] > 10**12, "timestamp"] = df.loc[df["timestamp"] > 10**12, "timestamp"] // 1000

merged = pd.concat([df18, df23], ignore_index=True)

merged = merged.drop_duplicates(subset=["user_id", "parent_asin", "timestamp"])

merged = merged.sort_values(by=["user_id", "timestamp", "parent_asin"]).reset_index(drop=True)

merged.to_csv(output_path, index=False, encoding="utf-8-sig")

print("saving:", output_path)
print(merged.head())

import pandas as pd

path_table2_2018 = "/content/drive/MyDrive/assignment/table2_metadata_aligned_to_table1_2018.csv"
path_table2_2023 = "/content/drive/MyDrive/assignment/table2_metadata_aligned_to_table1_2023.csv"
output_table2 = "/content/drive/MyDrive/assignment/table2_metadata.csv"

df18_t2 = pd.read_csv(path_table2_2018)
df23_t2 = pd.read_csv(path_table2_2023)

if "parent_asin" in df23_t2.columns and "asin" not in df23_t2.columns:
    df23_t2 = df23_t2.rename(columns={"parent_asin": "asin"})

drop_cols_2018 = ["image_source"]
drop_cols_2023 = ["main_category", "categories", "store", "image_source"]

existing_drop_cols_2018 = [c for c in drop_cols_2018 if c in df18_t2.columns]
existing_drop_cols_2023 = [c for c in drop_cols_2023 if c in df23_t2.columns]

df18_t2 = df18_t2.drop(columns=existing_drop_cols_2018)
df23_t2 = df23_t2.drop(columns=existing_drop_cols_2023)


df18_t2["asin"] = df18_t2["asin"].astype(str).str.strip()
df23_t2["asin"] = df23_t2["asin"].astype(str).str.strip()

common_cols = [c for c in df18_t2.columns if c in df23_t2.columns]

df18_t2_common = df18_t2[common_cols].copy()
df23_t2_common = df23_t2[common_cols].copy()

merged_t2 = pd.concat([df18_t2_common, df23_t2_common], ignore_index=True)
merged_t2 = merged_t2.drop_duplicates(subset=["asin"]).reset_index(drop=True)
merged_t2.to_csv(output_table2, index=False, encoding="utf-8-sig")

print("saving：", output_table2)
print(merged_t2.head())
