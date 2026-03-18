

!pip -q install pandas tqdm pillow requests

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
TABLE2_PATH = Path('/content/drive/MyDrive/assignment/table2_metadata.csv')

FEAT_DIR = Path('/content/drive/MyDrive/assignment/table2_resnet50')
FEAT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_IMG_DIR = Path('/content/drive/MyDrive/assignment/table2_metadata_images')
CACHE_IMG_DIR.mkdir(parents=True, exist_ok=True)


print("TABLE2_PATH :", TABLE2_PATH)
print("FEAT_DIR    :", FEAT_DIR)
print("TABLE exists?", TABLE2_PATH.exists())

# =========================
# Cell 4: Read tables + automatically identify fields
# =========================
import pandas as pd

df = pd.read_csv(TABLE2_PATH)

print(" shape:", df.shape)
print("column:", list(df.columns))
display(df.head(3))

item_id_candidates = ['asin', 'item_id', 'product_id', 'sku']
image_url_candidates = ['image_url', 'image_link', 'img_url', 'url']

item_col = next((c for c in item_id_candidates if c in df.columns), None)
img_col = next((c for c in image_url_candidates if c in df.columns), None)

if item_col is None:
    raise ValueError(f"The Product ID column was not found. Please ensure that one of these columns exists: {item_id_candidates}")
if img_col is None:
    raise ValueError(f"The image link column was not found. Please ensure that one of these columns exists: {image_url_candidates}")

print(f"Product ID column identified: {item_col}")
print(f"Image link column detected: {img_col}")

# Clean and construct the input table (keeping only one image per product).
work_df = df[[item_col, img_col]].copy()
work_df = work_df.rename(columns={item_col: 'item_id', img_col: 'image_url'})

work_df['item_id'] = work_df['item_id'].astype(str).str.strip()
work_df['image_url'] = work_df['image_url'].astype(str).str.strip()

invalid_strs = {'', 'nan', 'none', 'null', '[]'}
work_df = work_df[
    work_df['item_id'].notna() &
    work_df['image_url'].notna() &
    (~work_df['item_id'].str.lower().isin(invalid_strs)) &
    (~work_df['image_url'].str.lower().isin(invalid_strs))
].copy()

work_df = work_df[work_df['image_url'].str.startswith(('http://', 'https://'))].copy()

work_df = work_df.drop_duplicates(subset=['item_id'], keep='first').reset_index(drop=True)

print("\nNumber of characteristic products that can be extracted after cleaning:", len(work_df))
display(work_df.head(5))

# =========================
# Cell 5: Load the pre-trained ResNet50 (feature extraction)
# =========================
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

# Pre-trained weights
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)

# Remove the final classification layer and retain 2048-dimensional features.
model.fc = nn.Identity()
model = model.to(device).eval()

preprocess = weights.transforms()

print("ResNet50 Loaded, output visual feature dimension is approximately 2048")

# =========================
# Cell 6: Utility functions for image loading, caching, and batch feature extraction
# =========================
import io
import hashlib
import requests
import numpy as np
import torch

from pathlib import Path
from PIL import Image, UnidentifiedImageError

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Colab CNN Feature Extractor)"
})

def safe_filename_from_url(url: str, item_id: str, ext: str = ".jpg") -> str:
    """Generate a stable cache filename from item_id and URL hash."""
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    safe_item = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(item_id))[:80]
    return f"{safe_item}_{url_hash}{ext}"

def load_image_from_url(url: str, item_id: str, cache_dir: Path = None, timeout: int = 10):
    """
    Load an image from cache or download it from the URL.
    Returns:
        (PIL.Image in RGB mode, source)
    source:
        'cache' or 'download'
    """
    cache_path = None

    if cache_dir is not None:
        cache_path = cache_dir / safe_filename_from_url(url, item_id)
        if cache_path.exists():
            try:
                img = Image.open(cache_path).convert("RGB")
                return img, "cache"
            except Exception:
                try:
                    cache_path.unlink()
                except Exception:
                    pass

    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        raise ValueError(f"Invalid content type: {content_type}")

    img = Image.open(io.BytesIO(response.content)).convert("RGB")

    if cache_path is not None:
        try:
            img.save(cache_path, format="JPEG", quality=90)
        except Exception:
            pass

    return img, "download"

@torch.no_grad()
def extract_features_batch(records, model, preprocess, device):
    """
    Args:
        records: list of dicts, each containing:
            {'item_id': ..., 'image_url': ...}

    Returns:
        success_meta: list of metadata for successfully processed images
        feats_np: NumPy array of shape [B, D], or None if no image succeeded
        fail_rows: list of failed records with error messages
    """
    tensors = []
    success_meta = []
    fail_rows = []

    for rec in records:
        item_id = rec["item_id"]
        image_url = rec["image_url"]

        try:
            img, img_source = load_image_from_url(
                image_url,
                item_id,
                cache_dir=CACHE_IMG_DIR,
                timeout=10
            )
            x = preprocess(img)
            tensors.append(x)
            success_meta.append({
                "item_id": item_id,
                "image_url": image_url,
                "img_source": img_source
            })
        except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as e:
            fail_rows.append({
                "item_id": item_id,
                "image_url": image_url,
                "error": str(e)[:300]
            })

    if not tensors:
        return [], None, fail_rows

    batch = torch.stack(tensors, dim=0).to(device)
    feats = model(batch)

    if feats.ndim > 2:
        feats = torch.flatten(feats, 1)

    feats_np = feats.detach().cpu().numpy().astype("float32")
    return success_meta, feats_np, fail_rows

# =========================
# Cell 7: Batch visual feature extraction
# =========================
from math import ceil
import time
from tqdm.auto import tqdm

DEBUG_N = None
BATCH_SIZE = 64 if torch.cuda.is_available() else 16

run_df = work_df.copy()
if DEBUG_N is not None:
    run_df = run_df.head(DEBUG_N).copy()
    print(f"Debug mode: processing only the first {len(run_df)} items")

records = run_df[['item_id', 'image_url']].to_dict(orient='records')
num_total = len(records)

all_meta = []
all_feats = []
all_fails = []

start = time.time()

for i in tqdm(
    range(0, num_total, BATCH_SIZE),
    total=ceil(num_total / BATCH_SIZE),
    desc="Extracting CNN features"
):
    batch_records = records[i:i + BATCH_SIZE]
    success_meta, feats_np, fail_rows = extract_features_batch(
        batch_records, model, preprocess, device
    )

    all_fails.extend(fail_rows)

    if feats_np is not None and success_meta:
        all_meta.extend(success_meta)
        all_feats.append(feats_np)

elapsed = time.time() - start

if not all_feats:
    raise RuntimeError("No features were extracted successfully. Please check whether the image URLs are accessible.")

feature_matrix = np.concatenate(all_feats, axis=0)

print("\n========== Extraction Completed ==========")
print("Number of input items:", num_total)
print("Number of successful extractions:", len(all_meta))
print("Number of failed extractions:", len(all_fails))
print("Success rate:", f"{len(all_meta) / num_total:.2%}" if num_total else "N/A")
print("Feature matrix shape:", feature_matrix.shape)
print("Elapsed time (s):", round(elapsed, 2))

# =========================
# Cell 8: Save the offline feature library (Google Drive)
# =========================
from datetime import datetime
import json

# 1) Raw feature matrix (N, 2048)
feat_npy_path = FEAT_DIR / 'item_features_resnet50.npy'
np.save(feat_npy_path, feature_matrix)


# 3) Index mapping table: row index -> item_id / image_url
index_df = pd.DataFrame(all_meta).copy()
index_df.insert(0, "row_idx", range(len(index_df)))
index_csv_path = FEAT_DIR / 'item_feature_index.csv'
index_df.to_csv(index_csv_path, index=False, encoding='utf-8')

# 4) Failed image list (for retry or inspection)
failed_df = pd.DataFrame(all_fails)
fail_csv_path = FEAT_DIR / 'failed_images.csv'
failed_df.to_csv(fail_csv_path, index=False, encoding='utf-8')

# 5) Metadata (model, parameters, success rate, etc.)
meta_info = {
    "task": "cnn_visual_feature_extraction",
    "model_name": "torchvision_resnet50_imagenet1k_v2_fc_identity",
    "feature_dim": int(feature_matrix.shape[1]),
    "num_total_items_input": int(num_total),
    "num_success_features": int(len(all_meta)),
    "num_failed_images": int(len(all_fails)),
    "success_rate": float(len(all_meta) / num_total) if num_total else 0.0,
    "batch_size": int(BATCH_SIZE),
    "device": str(device),
    "source_table_path": str(TABLE2_PATH),
    "cache_image_dir": str(CACHE_IMG_DIR),
    "created_at_utc": datetime.utcnow().isoformat() + "Z"
}
meta_json_path = FEAT_DIR / 'feature_meta.json'
with open(meta_json_path, 'w', encoding='utf-8') as f:
    json.dump(meta_info, f, ensure_ascii=False, indent=2)

print(" Offline feature library saved successfully:")
print("-", feat_npy_path)
print("-", index_csv_path)
print("-", fail_csv_path)
print("-", meta_json_path)
