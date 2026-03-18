"""Main pipeline for product recommendation generation."""
import os
import re
import shutil
import textwrap
import pandas as pd
from collections import Counter
from llm_assistant import generate_recommendation

# Read terminal width so CLI output can be wrapped neatly.
def get_terminal_width(default_width: int = 100) -> int:
    try:
        return shutil.get_terminal_size(fallback=(default_width, 24)).columns
    except Exception:
        return default_width

# Wrap long CLI text while preserving numbered and bulleted formatting.
def wrap_text_preserve_format(text: str, width: int = None) -> str:
    if width is None:
        width = max(40, get_terminal_width() - 2)

    wrapped_lines = []
    for line in text.splitlines():
        if not line.strip():
            wrapped_lines.append("")
            continue

        leading_spaces = len(line) - len(line.lstrip(" "))
        indent = " " * leading_spaces
        stripped = line.lstrip(" ")

        if re.match(r"^\d+\.\s+", stripped):
            prefix_match = re.match(r"^(\d+\.\s+)", stripped)
            prefix = prefix_match.group(1)
            content = stripped[len(prefix):]
            wrapped = textwrap.fill(
                content,
                width=width,
                initial_indent=indent + prefix,
                subsequent_indent=indent + " " * len(prefix),
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_lines.append(wrapped)
            continue

        if re.match(r"^-\s+", stripped):
            prefix = "- "
            content = stripped[len(prefix):]
            wrapped = textwrap.fill(
                content,
                width=width,
                initial_indent=indent + prefix,
                subsequent_indent=indent + " " * len(prefix),
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_lines.append(wrapped)
            continue

        wrapped = textwrap.fill(
            stripped,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.append(wrapped)
    return "\n".join(wrapped_lines)

# Load product catalog data with multiple encoding fallbacks.
def load_table2(table2_path: str) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(table2_path, encoding=enc)
            print(f"table2 loaded successfully with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError("Unable to read table2_metadata_merged.csv, please check the file encoding or format.")

    # Remove CSV export artifacts such as unnamed index columns.
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # The app now generates introduction text dynamically, so old llm_text is ignored.
    if "llm_text" in df.columns:
        df = df.drop(columns=["llm_text"])

    for col in ["asin", "title", "description", "price", "image_url"]:
        if col not in df.columns:
            df[col] = None

    df["asin"] = df["asin"].astype(str).str.strip()
    return df

# Load user history / prediction vectors with encoding fallback.
def load_vector(vector_path: str) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(vector_path, encoding=enc)
            print(f"vector file loaded successfully with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError("Unable to read vector_llm_tasks.csv, please check the file encoding or format.")

    required_cols = ["user_id", "history_asin_list", "pred_asin", "pred_title"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    return df[required_cols]

# Return a clean string and convert NaN to empty text.
def safe_cell_value(row, col_name: str) -> str:
    value = row.get(col_name, "")
    if pd.isna(value):
        return ""
    return str(value).strip()

# Normalize price text for display without forcing numeric conversion.
def format_price_for_display(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text

# Normalize ASINs so matching is robust to punctuation and spacing.
def normalize_asin(asin: str) -> str:
    asin = str(asin).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", asin)

# Split a text field that may contain multiple ASINs.
def split_asin_text(asin_text: str):
    if not asin_text or not str(asin_text).strip():
        return []
    asin_list = re.split(r"[,|\s;]+", str(asin_text).strip())
    return [x.strip() for x in asin_list if x.strip()]

# Expand ASINs into a readable product context block for prompting or debugging.
def build_product_descriptions_from_asins(asin_text: str, table2_df: pd.DataFrame, max_items: int = None) -> str:
    asin_list = split_asin_text(asin_text)
    if max_items is not None:
        asin_list = asin_list[:max_items]

    results = []
    used = set()
    normalized_asin_series = table2_df["asin"].astype(str).apply(normalize_asin)
    for asin in asin_list:
        asin_norm = normalize_asin(asin)
        if not asin_norm or asin_norm in used:
            continue
        used.add(asin_norm)

        matched = table2_df[normalized_asin_series == asin_norm]
        if matched.empty:
            continue

        row = matched.iloc[0]
        title = safe_cell_value(row, "title")
        price = format_price_for_display(row.get("price", ""))
        desc = safe_cell_value(row, "description")
        results.append(
            f"ASIN: {safe_cell_value(row, 'asin')}\n"
            f"Title: {title}\n"
            f"Price: {price}\n"
            f"Description: {desc}"
        )
    return "\n\n".join(results)

# Tokenize text into lightweight keywords for matching and scoring.
def tokenize_text(text: str):
    text = str(text).lower()
    words = re.findall(r"[a-z0-9]+", text)
    stop_words = {
        "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with",
        "is", "are", "this", "that", "it", "be", "as", "by", "from", "at",
        "user", "product", "products", "description", "descriptions",
        "recommend", "recommended", "please", "suitable", "need", "needs",
        "want", "looking", "look", "something", "good", "best", "help",
    }
    return [w for w in words if len(w) > 2 and w not in stop_words]

# Build unigram, bigram, and trigram phrases so phrase matches can be rewarded.
def extract_query_phrases(text: str):
    words = tokenize_text(text)
    phrases = set(words)
    phrases.update(" ".join(words[i:i + 2]) for i in range(len(words) - 1))
    phrases.update(" ".join(words[i:i + 3]) for i in range(len(words) - 2))
    return {p for p in phrases if p.strip()}

# Score a product row against behavior keywords and explicit query phrases.
def score_text_matches(row_text: str, keyword_counter: Counter, query_phrases=None) -> int:
    row_text = row_text.lower()
    score = 0
    for word, weight in keyword_counter.items():
        if word in row_text:
            score += weight

    if query_phrases:
        for phrase in query_phrases:
            if phrase and phrase in row_text:
                score += 18 if " " in phrase else 6
    return score

# Detect simple low-price intent from the current query.
def _is_price_sensitive(user_query: str) -> bool:
    q = str(user_query).lower()
    price_words = ["cheap", "budget", "affordable", "low price", "lower price", "cheaper", "inexpensive"]
    return any(x in q for x in price_words)

# Best-effort numeric conversion used for price-sensitive ranking.
def _safe_float(value):
    try:
        text = str(value).strip().replace("$", "").replace(",", "")
        if not text or text.lower() == "nan":
            return None
        return float(text)
    except Exception:
        return None

# Return structured product dictionaries for a set of ASINs.
def get_products_from_asins(asin_text: str, table2_df: pd.DataFrame, max_items: int = None) -> list[dict]:
    asin_list = split_asin_text(asin_text)
    if max_items is not None:
        asin_list = asin_list[:max_items]

    results = []
    used = set()
    normalized_asin_series = table2_df["asin"].astype(str).apply(normalize_asin)
    for asin in asin_list:
        asin_norm = normalize_asin(asin)
        if not asin_norm or asin_norm in used:
            continue
        matched = table2_df[normalized_asin_series == asin_norm]
        if matched.empty:
            continue
        used.add(asin_norm)
        row = matched.iloc[0]
        results.append(
            {
                "asin": safe_cell_value(row, "asin"),
                "title": safe_cell_value(row, "title"),
                "price": format_price_for_display(row.get("price", "")),
                "description": safe_cell_value(row, "description"),
                "image_url": safe_cell_value(row, "image_url"),
            }
        )
    return results

# Build a weighted keyword profile from purchase history and predicted interests.
def build_behavior_keyword_counter(history_products: list[dict], predicted_products: list[dict], pred_title_text: str = "") -> Counter:
    counter = Counter()
    for product in history_products:
        for word in tokenize_text(f"{product.get('title', '')} {product.get('description', '')}"):
            counter[word] += 2
    for product in predicted_products:
        for word in tokenize_text(f"{product.get('title', '')} {product.get('description', '')}"):
            counter[word] += 3
    for word in tokenize_text(pred_title_text):
        counter[word] += 4
    return counter

""" Build a local shortlist from the full table2 catalog.
    Current user requirement is the strongest signal.
    Historical purchases and vector predictions are supporting signals."""
def get_shortlisted_products(
    table2_df: pd.DataFrame,
    user_query: str,
    historical_context: str,
    predicted_context: str,
    shortlist_size: int = 8,
    history_asin_text: str = "",
    predicted_asin_text: str = "",
    pred_title_text: str = "",
):
    history_products = get_products_from_asins(history_asin_text, table2_df, max_items=20)
    predicted_products = get_products_from_asins(predicted_asin_text, table2_df, max_items=10)

    query_words = tokenize_text(user_query)
    query_phrases = extract_query_phrases(user_query)
    behavior_counter = build_behavior_keyword_counter(history_products, predicted_products, pred_title_text=pred_title_text)

    history_asins = {normalize_asin(x) for x in split_asin_text(history_asin_text)}
    predicted_asins = {normalize_asin(x) for x in split_asin_text(predicted_asin_text)}
    price_sensitive = _is_price_sensitive(user_query)

    scored_rows = []
    seen_asins = set()

    for _, row in table2_df.iterrows():
        asin = safe_cell_value(row, "asin")
        asin_norm = normalize_asin(asin)
        if not asin_norm or asin_norm in seen_asins:
            continue
        seen_asins.add(asin_norm)

        title = safe_cell_value(row, "title")
        price = format_price_for_display(row.get("price", ""))
        description = safe_cell_value(row, "description")
        image_url = safe_cell_value(row, "image_url")

        row_title = title.lower()
        row_desc = description.lower()
        row_text = f"{row_title} {row_desc}"

        # Query relevance is the main signal because the system should respond to the current need.
        title_hits = sum(1 for w in query_words if w in row_title)
        desc_hits = sum(1 for w in query_words if w in row_desc)
        phrase_hits = sum(1 for p in query_phrases if p and p in row_text)

        query_score = title_hits * 40 + desc_hits * 16 + phrase_hits * 28
        behavior_score = score_text_matches(row_text, behavior_counter)

        # Penalize rows that do not match the current requirement at all.
        if query_words:
            if title_hits == 0 and desc_hits == 0 and phrase_hits == 0:
                query_score -= 120
            else:
                query_score += 40

        # Small boosts for overlap with historical and predicted ASINs.
        if asin_norm in history_asins:
            behavior_score += 16
        if asin_norm in predicted_asins:
            behavior_score += 22

        price_value = _safe_float(price)
        price_score = 0
        if price_sensitive and price_value is not None:
            price_score += max(0, int(35 - min(price_value, 35)))
        elif price_sensitive and price_value is None:
            price_score -= 5

        # Prefer rows with richer display assets when scores are otherwise similar.
        has_image_bonus = 4 if image_url.strip() else 0
        has_desc_bonus = 4 if description.strip() else 0

        score = query_score * 5 + behavior_score + price_score + has_image_bonus + has_desc_bonus
        scored_rows.append(
            {
                "score": score,
                "query_score": query_score,
                "behavior_score": behavior_score,
                "asin": asin,
                "asin_norm": asin_norm,
                "title": title,
                "price": price,
                "description": description,
                "image_url": image_url,
            }
        )

    # Prefer products that actually match the current requirement.
    scored_rows.sort(
        key=lambda x: (
            -(1 if x["query_score"] > 0 else 0),
            -x["query_score"],
            -x["score"],
            x["title"].lower(),
            x["asin_norm"],
        )
    )
    return scored_rows[:shortlist_size]

# Convert shortlisted rows into a prompt-friendly catalog block.
def build_shortlisted_catalog_context_from_products(shortlisted_products) -> str:
    lines = []
    for i, item in enumerate(shortlisted_products, start=1):
        lines.append(
            f"[Catalog Product {i}]\n"
            f"ASIN: {item['asin']}\n"
            f"Title: {item['title']}\n"
            f"Price: {item['price']}\n"
            f"Description: {item['description']}"
        )
    return "\n\n".join(lines)

# Command-line entry point for quick local testing.
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    table2_path = os.path.join(base_dir, "table2_metadata_merged.csv")
    vector_path = os.path.join(base_dir, "vector_llm_tasks.csv")

    print("Loading data...")
    table2_df = load_table2(table2_path)
    vector_df = load_vector(vector_path)
    print(f"table2 rows: {len(table2_df)} | vector rows: {len(vector_df)}")
    user_input = input("Enter user row number or user ID: ").strip()
    if not user_input:
        user_row = vector_df.iloc[0]
    elif user_input.isdigit():
        idx = int(user_input)
        if idx >= len(vector_df):
            print("Row number out of range, using default user 0")
            idx = 0
        user_row = vector_df.iloc[idx]
    else:
        matched = vector_df[vector_df["user_id"].astype(str) == user_input]
        user_row = vector_df.iloc[0] if matched.empty else matched.iloc[0]

    user_id = str(user_row.get("user_id", "") or "")
    history_asin_text = user_row.get("history_asin_list", None)
    pred_asin_text = user_row.get("pred_asin", None)
    pred_title_text = str(user_row.get("pred_title", "") or "")

    history_description_text = build_product_descriptions_from_asins(history_asin_text, table2_df, max_items=None)
    predicted_description_text = build_product_descriptions_from_asins(pred_asin_text, table2_df, max_items=None)

    user_query = input("Enter the user's current requirement (e.g. recommend a cheaper product): ").strip()
    if not user_query:
        user_query = "Please recommend suitable products for me."

    shortlisted_products = get_shortlisted_products(
        table2_df=table2_df,
        user_query=user_query,
        historical_context=history_description_text,
        predicted_context=predicted_description_text,
        shortlist_size=8,
        history_asin_text=history_asin_text,
        predicted_asin_text=pred_asin_text,
        pred_title_text=pred_title_text,
    )
    shortlisted_catalog_context = build_shortlisted_catalog_context_from_products(shortlisted_products)

    try:
        result = generate_recommendation(
            user_id=user_id,
            user_query=user_query,
            historical_context=history_description_text,
            predicted_context=predicted_description_text,
            shortlisted_catalog_context=shortlisted_catalog_context,
            use_gemini=True,
        )
        print("\n====== Guide recommendation output ======\n")
        print(wrap_text_preserve_format(result))
        print("\n=========================\n")
    except Exception as e:
        print("\n====== Guide recommendation output ======\n")
        print(wrap_text_preserve_format(str(e)))
        print("\n=========================\n")

if __name__ == "__main__":
    main()