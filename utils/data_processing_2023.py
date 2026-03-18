
!pip -q install pandas pyarrow tqdm

from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

# =========================
# Cell 2
# =========================
import os

RAW_REVIEW_FILES = [
    "/content/drive/MyDrive/assignment/All_Beauty23.jsonl (1)",
]
OUTPUT_DIR = "/content/drive/MyDrive/assignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 1
DROP_DUPLICATES = True

# =========================
# Cell 3: Generate user interaction behavior table
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
    First try standard JSON parsing;
    if it fails, try ast.literal_eval for compatibility with older formats.
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

        # Roughly count total lines first (optional, for progress display)
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

                # ===== New field names =====
                user_id = obj.get("user_id")
                parent_asin = obj.get("parent_asin")
                timestamp = obj.get("timestamp")

                # Fallback: some datasets may use "time"
                if timestamp is None:
                    timestamp = obj.get("time")

                if user_id is None or parent_asin is None or timestamp is None:
                    missing_fields += 1
                    continue

                # Convert timestamp to int (new versions often use millisecond timestamps)
                try:
                    timestamp = int(float(timestamp))
                except Exception:
                    continue

                # If it is a millisecond timestamp, convert it to seconds for consistency
                if timestamp > 10**12:
                    timestamp = timestamp // 1000

                # Basic validity check
                if timestamp <= 0:
                    continue

                rows.append((str(user_id), str(parent_asin), timestamp))

    if not rows:
        raise ValueError("No interaction records were successfully extracted. Please check the file paths or field names.")

    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "timestamp"])

    print("\n========== Raw Extraction Statistics ==========")
    print(f"Extracted interaction records: {len(df):,}")
    print(f"Failed parsed lines:          {bad_lines:,}")
    print(f"Lines with missing fields:   {missing_fields:,}")

    # Remove duplicates to avoid repeated review records affecting sequence modeling
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["user_id", "parent_asin", "timestamp"]).copy()
        after = len(df)
        print(f"Records after deduplication: {after:,} (removed {before - after:,} duplicates)")

    # Filter low-activity users
    if min_user_interactions > 1:
        user_cnt = df["user_id"].value_counts()
        keep_users = user_cnt[user_cnt >= min_user_interactions].index
        before = len(df)
        df = df[df["user_id"].isin(keep_users)].copy()
        after = len(df)
        print(f"Records after user filtering: {after:,} (removed {before - after:,}, threshold={min_user_interactions})")

    # Filter low-frequency items (optional)
    if min_item_interactions > 1:
        item_cnt = df["parent_asin"].value_counts()
        keep_items = item_cnt[item_cnt >= min_item_interactions].index
        before = len(df)
        df = df[df["parent_asin"].isin(keep_items)].copy()
        after = len(df)
        print(f"Records after item filtering: {after:,} (removed {before - after:,}, threshold={min_item_interactions})")

    # Sort by user + time (important for LSTM training)
    df = df.sort_values(["user_id", "timestamp", "parent_asin"]).reset_index(drop=True)

    return df

# Run
table1_23 = build_table1(
    RAW_REVIEW_FILES,
    min_user_interactions=MIN_USER_INTERACTIONS,
    min_item_interactions=MIN_ITEM_INTERACTIONS,
    drop_duplicates=DROP_DUPLICATES
)

print("\n========== Final Table 1 Statistics ==========")
print(f"Total interactions: {len(table1_23):,}")
print(f"Number of users:    {table1_23['user_id'].nunique():,}")
print(f"Number of items:    {table1_23['parent_asin'].nunique():,}")
print("\nTable 1 Preview:")
display(table1_23.head(10))

# =========================
# Cell 4: Save table
# =========================
csv_path = os.path.join(OUTPUT_DIR, "table1_interactions_2023.csv")
table1_23.to_csv(csv_path, index=False)
print(csv_path)

# =========================
# Cell 5:
# =========================
import os
RAW_META_FILES = [
    "/content/drive/MyDrive/assignment/meta_23.jsonl (1)",
]

OUTPUT_DIR = "/content/drive/MyDrive/assignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)
STRICT_REQUIRE_BOTH_TITLE_AND_DESC = True
DROP_DUPLICATES_BY_ASIN = True

# =========================
# Cell 6: Build the product metadata table (Table 2)
# Purpose:
# Parse the raw metadata file(s), extract product text/image/price
# fields, remove incomplete or unusable records, keep the best
# record for each product, and create a structured metadata table
# for image-based retrieval and LLM use.
# =========================
import re
import math


def is_valid_url(x):
    """Check whether a string is a valid HTTP/HTTPS URL."""
    if not isinstance(x, str):
        return False
    x = x.strip()
    return x.startswith("http://") or x.startswith("https://")


def normalize_text(x):
    """Normalize title/description/features fields into clean strings.

    Common input types:
    - string
    - list of strings
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

    # Remove repeated whitespace.
    x = re.sub(r"\s+", " ", x).strip()

    # Treat placeholder values as missing.
    if x == "" or x.lower() in {"nan", "none", "null", "[]"}:
        return None
    return x


def parse_price(price):
    """Parse a price value and return float or None.

    Supported examples:
    - 12.99
    - "$12.99"
    - "12,99" / "$1,299.00"
    - None
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

    # Handle thousands separators and decimal symbols.
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", "")

    try:
        return float(s)
    except Exception:
        return None


def extract_image_url_hires_or_large(meta_obj):
    """Extract the best available image URL from the `images` field.

    Priority:
    1) MAIN hi_res
    2) MAIN large
    3) Any hi_res
    4) Any large

    Returns:
    - image_url
    - image_source
    """
    images = meta_obj.get("images")
    if not isinstance(images, list):
        return None, None

    # 1) MAIN hi_res
    for item in images:
        if not isinstance(item, dict):
            continue
        variant = str(item.get("variant", "")).strip().upper()
        hi_res = item.get("hi_res")
        if variant == "MAIN" and is_valid_url(hi_res):
            return hi_res.strip(), "images.hi_res(MAIN)"

    # 2) MAIN large
    for item in images:
        if not isinstance(item, dict):
            continue
        variant = str(item.get("variant", "")).strip().upper()
        large = item.get("large")
        if variant == "MAIN" and is_valid_url(large):
            return large.strip(), "images.large(MAIN)"

    # 3) Any hi_res
    for item in images:
        if not isinstance(item, dict):
            continue
        hi_res = item.get("hi_res")
        if is_valid_url(hi_res):
            return hi_res.strip(), "images.hi_res"

    # 4) Any large
    for item in images:
        if not isinstance(item, dict):
            continue
        large = item.get("large")
        if is_valid_url(large):
            return large.strip(), "images.large"

    return None, None


def normalize_categories(x):
    """Convert categories into a clean string for analysis and display."""
    if x is None:
        return None
    if isinstance(x, list):
        parts = [str(i).strip() for i in x if str(i).strip()]
        return " > ".join(parts) if parts else None
    s = str(x).strip()
    return s if s else None


def build_table2_metadata(meta_files, drop_duplicates_by_asin=True):
    """Build the cleaned product metadata table.

    Expected output columns include:
    - parent_asin
    - title
    - description
    - price
    - image_url
    - image_source
    - main_category
    - categories
    - store
    - llm_text
    """
    rows = []
    bad_lines = 0
    missing_parent_asin = 0

    for fp in meta_files:
        print(f"\n[Reading metadata file] {fp}")
        if not os.path.exists(fp):
            print(f"  !! File not found, skipped: {fp}")
            continue

        # Count total lines so tqdm can show progress.
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

                # Core product ID field in the newer dataset format.
                parent_asin = obj.get("parent_asin")
                if parent_asin is None:
                    missing_parent_asin += 1
                    continue
                parent_asin = str(parent_asin).strip()
                if not parent_asin:
                    missing_parent_asin += 1
                    continue

                # Text fields.
                title = normalize_text(obj.get("title"))

                # Newer versions often store description as a list.
                description = normalize_text(obj.get("description"))

                # Fallback: use `features` when description is missing.
                if description is None:
                    description = normalize_text(obj.get("features"))

                # Price field.
                price = parse_price(obj.get("price"))

                # Prefer hi_res images, otherwise fall back to large images.
                image_url, image_source = extract_image_url_hires_or_large(obj)

                # Optional fields kept for downstream analysis.
                main_category = normalize_text(obj.get("main_category"))
                categories = normalize_categories(obj.get("categories"))
                store = normalize_text(obj.get("store"))

                rows.append({
                    "parent_asin": parent_asin,
                    "title": title,
                    "description": description,
                    "price": price,
                    "image_url": image_url,
                    "image_source": image_source,
                    "main_category": main_category,
                    "categories": categories,
                    "store": store,
                })

    if not rows:
        raise ValueError("No metadata records were extracted. Please check the file paths or the data format.")

    df = pd.DataFrame(rows)

    print("\n========== Raw Metadata Extraction Statistics ==========")
    print(f"Metadata records:               {len(df):,}")
    print(f"Failed parsed lines:           {bad_lines:,}")
    print(f"Rows missing parent_asin:      {missing_parent_asin:,}")

    # Remove products with missing price.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    before = len(df)
    df = df[df["price"].notna()].copy()
    print(f"After price filtering:         {len(df):,} (removed {before - len(df):,} records without price)")

    # Keep only products with a usable image.
    before = len(df)
    has_img = df["image_url"].notna() & (df["image_url"].astype(str).str.strip() != "")
    df = df[has_img].copy()
    print(f"After image filtering:         {len(df):,} (removed {before - len(df):,} records without usable images)")

    # Keep records with at least one usable text field.
    before = len(df)
    has_title = df["title"].notna() & (df["title"].astype(str).str.strip() != "")
    has_desc = df["description"].notna() & (df["description"].astype(str).str.strip() != "")
    df = df[has_title | has_desc].copy()
    print(f"After text filtering:          {len(df):,} (removed {before - len(df):,} records with neither title nor description)")

    # Fine-grained text availability statistics.
    has_title = df["title"].notna() & (df["title"].astype(str).str.strip() != "")
    has_desc = df["description"].notna() & (df["description"].astype(str).str.strip() != "")
    only_title = (has_title & (~has_desc)).sum()
    only_desc = ((~has_title) & has_desc).sum()
    both_text = (has_title & has_desc).sum()
    print(f"  - Title only:                {only_title:,}")
    print(f"  - Description only:          {only_desc:,}")
    print(f"  - Both title and description:{both_text:,}")

    # Fallback LLM text: prefer description, otherwise use title.
    df["llm_text"] = df["description"].where(
        df["description"].notna() & (df["description"].astype(str).str.strip() != ""),
        df["title"]
    )

    # Deduplicate by parent_asin and keep the most informative record.
    if drop_duplicates_by_asin:
        before = len(df)

        df["title_len"] = df["title"].fillna("").astype(str).str.len()
        df["desc_len"] = df["description"].fillna("").astype(str).str.len()
        df["llm_text_len"] = df["llm_text"].fillna("").astype(str).str.len()

        # Prefer hi_res over large during deduplication.
        df["img_quality_score"] = 0
        df.loc[df["image_source"].astype(str).str.contains("hi_res", na=False), "img_quality_score"] = 2
        df.loc[df["image_source"].astype(str).str.contains("large", na=False), "img_quality_score"] = 1

        # Higher score means a better record to keep.
        df["keep_score"] = (
            df["img_quality_score"] * 100000
            + df["llm_text_len"] * 100
            + df["desc_len"] * 10
            + df["title_len"]
        )

        df = df.sort_values(["parent_asin", "keep_score"], ascending=[True, False])
        df = df.drop_duplicates(subset=["parent_asin"], keep="first").copy()

        # Drop helper columns used only for scoring.
        df = df.drop(columns=["title_len", "desc_len", "llm_text_len", "img_quality_score", "keep_score"])

        print(f"After deduplication by parent_asin: {len(df):,} (removed {before - len(df):,} duplicate records)")

    # Final sorting.
    df = df.sort_values("parent_asin").reset_index(drop=True)

    return df

# =========================
# Cell 7: Run Table 2 construction and inspect results
# Purpose:
# Build the metadata table, review summary statistics, and check
# the distribution of retained image sources.
# =========================
table2_23 = build_table2_metadata(
    RAW_META_FILES,
    drop_duplicates_by_asin=DROP_DUPLICATES_BY_ASIN
)

print("\n========== Final Table 2 Statistics ==========")
print(f"Number of products:            {len(table2_23):,}")
print(f"Unique products (parent_asin): {table2_23['parent_asin'].nunique():,}")

print("\nImage source distribution:")
print(table2_23["image_source"].value_counts(dropna=False))

print("\nTable 2 Preview:")
display(table2_23.head(10))

# =========================
# Cell 8: Align Table 2 with Table 1 (optional)
# Purpose:
# Keep only products that actually appear in the interaction table,
# so the metadata table matches the products used in sequence data.
# =========================
table1_path = "/content/drive/MyDrive/assignment/table1_interactions_2023.csv"  # Update this path if needed.
table1_df = None

if os.path.exists(table1_path):
    table1_df = pd.read_csv(table1_path, usecols=["parent_asin"])
    used_asins = set(table1_df["parent_asin"].astype(str).unique())

    before = len(table2_23)
    table2_23 = table2_23[table2_23["parent_asin"].astype(str).isin(used_asins)].copy().reset_index(drop=True)
    print(f"\nProducts after alignment with Table 1: {len(table2_23):,} (removed {before - len(table2_23):,} products not used in interactions)")

    # Save the aligned version.
    aligned_csv = os.path.join(OUTPUT_DIR, "table2_metadata_aligned_to_table1_2023.csv")

    table2_23.to_csv(aligned_csv, index=False)

    print("Aligned version saved:")
    print(aligned_csv)


    print("\nAligned Table 2 Preview:")
    display(table2_23.head(10))
else:
    print("\nTable 1 file not found. Alignment step skipped.")

