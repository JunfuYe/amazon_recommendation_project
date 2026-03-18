

!pip -q install pandas pyarrow tqdm

from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

# =========================
# Cell 2
# =========================
import os

RAW_REVIEW_FILES = [
    "/content/drive/MyDrive/assignment/All_Beauty_2018.json",
]
OUTPUT_DIR = "/content/drive/MyDrive/assignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 1
DROP_DUPLICATES = True

# =========================
# Cell 3: Build Table 1 (User Interaction Table)
# =========================
import os
import gzip
import json
import ast
import pandas as pd
from tqdm.auto import tqdm

def open_maybe_gz(path):
    """Support both .gz files and plain text files."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def parse_json_line(line):
    """
    First try to parse as standard JSON.
    If that fails, fall back to ast.literal_eval
    for compatibility with some older formats.
    """
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None

def build_table1(review_files, min_user_interactions=5, min_item_interactions=1, drop_duplicates=True):
    rows = []
    bad_lines = 0
    missing_fields = 0

    for fp in review_files:
        print(f"\n[Reading file] {fp}")
        if not os.path.exists(fp):
            print(f"  !! File does not exist, skipped: {fp}")
            continue

        # Roughly count the number of lines first (optional, useful for progress display)
        # This is usually acceptable even for large files; comment it out if too slow
        total_lines = 0
        with open_maybe_gz(fp) as f:
            for _ in f:
                total_lines += 1

        with open_maybe_gz(fp) as f:
            for line in tqdm(f, total=total_lines, desc=os.path.basename(fp)):
                obj = parse_json_line(line)
                if obj is None:
                    bad_lines += 1
                    continue

                # Target fields (modify here if your dataset uses different field names)
                reviewerID = obj.get("reviewerID")
                asin = obj.get("asin")
                unixReviewTime = obj.get("unixReviewTime")

                # Some datasets may use alternative time fields (fallback)
                if unixReviewTime is None:
                    unixReviewTime = obj.get("timestamp") or obj.get("time")

                if reviewerID is None or asin is None or unixReviewTime is None:
                    missing_fields += 1
                    continue

                # Convert timestamp to int
                try:
                    unixReviewTime = int(unixReviewTime)
                except Exception:
                    continue

                # Basic validity filtering
                if unixReviewTime <= 0:
                    continue

                rows.append((str(reviewerID), str(asin), unixReviewTime))

    if not rows:
        raise ValueError("No interaction records were successfully extracted. Please check the file paths or field names.")

    df = pd.DataFrame(rows, columns=["reviewerID", "asin", "unixReviewTime"])

    print("\n========== Raw Extraction Statistics ==========")
    print(f"Number of extracted interactions: {len(df):,}")
    print(f"Number of failed parsed lines:   {bad_lines:,}")
    print(f"Number of missing-field lines:  {missing_fields:,}")

    # Remove duplicates to avoid repeated review records interfering with sequence construction
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["reviewerID", "asin", "unixReviewTime"]).copy()
        after = len(df)
        print(f"Records after deduplication:    {after:,} (removed {before - after:,} duplicates)")

    # Filter low-activity users
    if min_user_interactions > 1:
        user_cnt = df["reviewerID"].value_counts()
        keep_users = user_cnt[user_cnt >= min_user_interactions].index
        before = len(df)
        df = df[df["reviewerID"].isin(keep_users)].copy()
        after = len(df)
        print(f"Records after user filtering:   {after:,} (removed {before - after:,}, threshold={min_user_interactions})")

    # Filter low-frequency items (optional)
    if min_item_interactions > 1:
        item_cnt = df["asin"].value_counts()
        keep_items = item_cnt[item_cnt >= min_item_interactions].index
        before = len(df)
        df = df[df["asin"].isin(keep_items)].copy()
        after = len(df)
        print(f"Records after item filtering:   {after:,} (removed {before - after:,}, threshold={min_item_interactions})")

    # Sort by user + time (critical for LSTM training)
    df = df.sort_values(["reviewerID", "unixReviewTime", "asin"]).reset_index(drop=True)

    return df

# Run
table1_df = build_table1(
    RAW_REVIEW_FILES,
    min_user_interactions=MIN_USER_INTERACTIONS,
    min_item_interactions=MIN_ITEM_INTERACTIONS,
    drop_duplicates=DROP_DUPLICATES
)

print("\n========== Table 1 (Final) Statistics ==========")
print(f"Total interactions: {len(table1_df):,}")
print(f"Number of users:    {table1_df['reviewerID'].nunique():,}")
print(f"Number of items:    {table1_df['asin'].nunique():,}")
print("\nPreview of Table 1:")
display(table1_df.head(10))

# =========================
# Cell 4: Save Table 1
# =========================
csv_path = os.path.join(OUTPUT_DIR, "table1_interactions_2018.csv")
table1_df.to_csv(csv_path, index=False)
print("Saved files:")
print(csv_path)

# =========================
# Cell 5: metadata processing
# =========================
import os
RAW_META_FILES = [
    "/content/drive/MyDrive/assignment/meta_All_Beauty_2018.json",

]
OUTPUT_DIR = "/content/drive/MyDrive/assignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)
STRICT_REQUIRE_BOTH_TITLE_AND_DESC = True
DROP_DUPLICATES_BY_ASIN = True

# =========================
# Cell 6: Process metadata and build Table 2
# =========================
import os
import gzip
import json
import ast
import re
import math
import pandas as pd
from tqdm.auto import tqdm

def open_maybe_gz(path):
    """Support both .gz files and plain text files."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def parse_json_line(line):
    """Parse one line as JSON, with ast.literal_eval as a fallback."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        try:
            return ast.literal_eval(line)  # Compatible with some older Amazon formats
        except Exception:
            return None

def is_valid_url(x):
    """Check whether a value is a valid HTTP/HTTPS URL."""
    if not isinstance(x, str):
        return False
    x = x.strip()
    return x.startswith("http://") or x.startswith("https://")

def normalize_text(x):
    """
    Clean title/description into a normal string.

    Common cases for description:
    - string
    - list (multiple text segments)
    - None
    """
    if x is None:
        return None

    if isinstance(x, list):
        parts = []
        for item in x:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                parts.append(s)
        if not parts:
            return None
        x = " ".join(parts)
    else:
        x = str(x).strip()

    # Remove extra whitespace
    x = re.sub(r"\s+", " ", x).strip()

    # Treat invalid placeholders as missing values
    if x == "" or x.lower() in {"nan", "none", "null", "[]"}:
        return None
    return x

def parse_price(price):
    """
    Price may be:
    - 12.99
    - "$12.99"
    - "12,99" / "$1,299.00"
    - None

    Return float or None.
    """
    if price is None:
        return None

    if isinstance(price, (int, float)):
        if isinstance(price, float) and (math.isnan(price) or math.isinf(price)):
            return None
        return float(price)

    s = str(price).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None

    s = s.replace("£", "").replace("$", "").replace("€", "").strip()

    # Handle thousand separators / decimal points
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", "")

    try:
        return float(s)
    except Exception:
        return None

def extract_image_url(meta_obj):
    """
    Prefer high-resolution image fields first, then fallback to normal image fields.
    Return: (image_url, image_source)
    """
    # 1) High-resolution image fields (common variants)
    candidate_highres_fields = ["high_res_images", "imageURLHighRes", "image_url_high_res"]
    for f in candidate_highres_fields:
        if f in meta_obj:
            v = meta_obj.get(f)

            # May be list[str] or list[dict]
            if isinstance(v, list):
                for item in v:
                    if is_valid_url(item):
                        return item.strip(), f
                    if isinstance(item, dict):
                        for k in ["url", "large", "hi_res", "link"]:
                            u = item.get(k)
                            if is_valid_url(u):
                                return u.strip(), f

            # May be a single string
            elif is_valid_url(v):
                return v.strip(), f

    # 2) Standard image fields (common variants)
    candidate_img_fields = ["imUrl", "image", "imageURL", "thumbnail"]
    for f in candidate_img_fields:
        if f in meta_obj:
            v = meta_obj.get(f)
            if is_valid_url(v):
                return v.strip(), f
            if isinstance(v, list):
                for item in v:
                    if is_valid_url(item):
                        return item.strip(), f

    return None, None

def build_table2_metadata(meta_files, drop_duplicates_by_asin=True):
    rows = []
    bad_lines = 0
    missing_asin = 0

    for fp in meta_files:
        print(f"\n[Reading metadata file] {fp}")
        if not os.path.exists(fp):
            print(f"  !! File does not exist, skipped: {fp}")
            continue

        # Count lines for tqdm progress bar
        total_lines = 0
        with open_maybe_gz(fp) as f:
            for _ in f:
                total_lines += 1

        with open_maybe_gz(fp) as f:
            for line in tqdm(f, total=total_lines, desc=os.path.basename(fp)):
                obj = parse_json_line(line)
                if obj is None:
                    bad_lines += 1
                    continue

                asin = obj.get("asin")
                if asin is None:
                    missing_asin += 1
                    continue
                asin = str(asin).strip()
                if not asin:
                    missing_asin += 1
                    continue

                title = normalize_text(obj.get("title"))

                # Some datasets use "description"; if missing, fall back to "feature"
                description = normalize_text(obj.get("description"))
                if description is None:
                    description = normalize_text(obj.get("feature"))

                price = parse_price(obj.get("price"))
                image_url, image_source = extract_image_url(obj)

                rows.append({
                    "asin": asin,
                    "title": title,
                    "description": description,
                    "price": price,
                    "image_url": image_url,
                    "image_source": image_source,
                })

    if not rows:
        raise ValueError("No metadata records were extracted. Please check the file paths or data format.")

    df = pd.DataFrame(rows)

    print("\n========== Raw Extraction Statistics ==========")
    print(f"Number of metadata records: {len(df):,}")
    print(f"Number of failed parsed lines: {bad_lines:,}")
    print(f"Number of missing asin lines:  {missing_asin:,}")

    # ---- Rule 1: Remove products without image links (minimum requirement for CNN) ----
    before = len(df)
    has_img = df["image_url"].notna() & (df["image_url"].astype(str).str.strip() != "")
    df = df[has_img].copy()
    print(f"After image filtering: {len(df):,} (removed {before - len(df):,} records without image links)")

    # ---- Rule 2 (relaxed): must have image + at least one of (title or description) ----
    before = len(df)

    has_title = df["title"].notna() & (df["title"].astype(str).str.strip() != "")
    has_desc  = df["description"].notna() & (df["description"].astype(str).str.strip() != "")

    # Relaxed condition: at least one of title or description must exist
    df = df[has_title | has_desc].copy()

    print(f"After text filtering (relaxed: at least one of title/description): {len(df):,} "
          f"(removed {before - len(df):,} records with neither title nor description)")

    # Optional: print more detailed statistics to explain the number changes
    # These statistics are more meaningful after image filtering
    has_title = df["title"].notna() & (df["title"].astype(str).str.strip() != "")
    has_desc  = df["description"].notna() & (df["description"].astype(str).str.strip() != "")
    only_title = (has_title & (~has_desc)).sum()
    only_desc = ((~has_title) & has_desc).sum()
    both_text = (has_title & has_desc).sum()
    print(f"  - Title only: {only_title:,}")
    print(f"  - Description only: {only_desc:,}")
    print(f"  - Both title and description: {both_text:,}")

    # ---- Add fallback field for LLM: prefer description, otherwise use title ----
    df["llm_text"] = df["description"].where(
        df["description"].notna() & (df["description"].astype(str).str.strip() != ""),
        df["title"]
    )

    # ---- Deduplication: if one asin has multiple records, keep the most informative one ----
    if drop_duplicates_by_asin:
        before = len(df)

        df["title_len"] = df["title"].fillna("").astype(str).str.len()
        df["desc_len"] = df["description"].fillna("").astype(str).str.len()
        df["llm_text_len"] = df["llm_text"].fillna("").astype(str).str.len()
        df["has_highres"] = (
            df["image_source"].astype(str).str.contains("high_res|imageURLHighRes", case=False, na=False)
        ).astype(int)

        # Higher score means higher priority to keep:
        # high-resolution image first, then longer llm_text, description, and title
        df["keep_score"] = (
            df["has_highres"] * 100000
            + df["llm_text_len"] * 100
            + df["desc_len"] * 10
            + df["title_len"]
        )

        df = df.sort_values(["asin", "keep_score"], ascending=[True, False])
        df = df.drop_duplicates(subset=["asin"], keep="first").copy()

        # Remove temporary columns
        df = df.drop(columns=["title_len", "desc_len", "llm_text_len", "has_highres", "keep_score"])

        print(f"After asin deduplication: {len(df):,} (removed {before - len(df):,} duplicate asin records)")

    # Final sorting
    df = df.sort_values("asin").reset_index(drop=True)

    return df

# =========================
# Cell 7:
# =========================
table2_df = build_table2_metadata(
    RAW_META_FILES,
    drop_duplicates_by_asin=DROP_DUPLICATES_BY_ASIN
)

print("\n========== Table 2 (Final) Statistics ==========")
print(f"Number of products: {len(table2_df):,}")
print(f"Number of products with price: {(table2_df['price'].notna()).sum():,}")
print("Distribution of image sources:")
print(table2_df['image_source'].value_counts(dropna=False))

display(table2_df.head(10))

# =========================
# Cell 8: Save Table 2：
# =========================
csv_path = os.path.join(OUTPUT_DIR, "table2_metadata_2018.csv")
table2_df.to_csv(csv_path, index=False)
print("\nsave：")
print(csv_path)

# =========================
# Cell 9 : Align with Table 1 and keep only items that appear in interactions
# =========================
table1_path = "/content/drive/MyDrive/assignment/table1_interactions_2018.csv"

if os.path.exists(table1_path):
    table1_df = pd.read_csv(table1_path, usecols=["asin"])
    used_asins = set(table1_df["asin"].astype(str).unique())

    before = len(table2_df)
    table2_df = table2_df[table2_df["asin"].astype(str).isin(used_asins)].copy().reset_index(drop=True)
    print(f"\nNumber of products after alignment with Table 1: {len(table2_df):,} "
          f"(removed {before - len(table2_df):,} items not appearing in interactions)")

    # Save the aligned version
    aligned_csv = os.path.join(OUTPUT_DIR, "table2_metadata_aligned_to_table1_2018.csv")

    table2_df.to_csv(aligned_csv, index=False)

    print("Aligned version saved:")
    print(aligned_csv)

    # Preview the aligned version
    display(table2_df.head(10))
else:
    print("\nTable 1 file not found. Alignment step skipped.")
