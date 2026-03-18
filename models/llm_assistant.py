"""Gemini helps with product recommendations."""
import json
import os
import re
from difflib import SequenceMatcher

try:
    from google import genai
except Exception:
    genai = None

# Read Gemini API key from environment variable or fill in the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or "YOUR_OWN_API_KEY"
client = genai.Client(api_key=GEMINI_API_KEY) if (GEMINI_API_KEY and genai is not None) else None

# Safely call Gemini and return plain text.
def _call_gemini(prompt: str, max_output_tokens: int = 900) -> str:
    if not client:
        return ""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"max_output_tokens": max_output_tokens},
        )
        return (response.text or "").strip()
    except Exception:
        return ""

# Recommendation generation is currently handled elsewhere.
# Keeping this stub avoids breaking the existing imports and call flow.
def generate_recommendation(
    user_id: str,
    user_query: str,
    historical_context: str,
    predicted_context: str,
    shortlisted_catalog_context: str,
    use_gemini: bool = True,
) -> str:
    return ""

# Collapse repeated whitespace so later comparisons are more stable.
def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()

# Normalize text before comparing generated copy and source descriptions.
def _normalize_for_similarity(text: str) -> str:
    text = _clean_text(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Reject generated copy that is too close to the catalog description.
def _looks_too_close_to_description(intro: str, description: str) -> bool:
    intro_n = _normalize_for_similarity(intro)
    desc_n = _normalize_for_similarity(description)
    if not intro_n or not desc_n:
        return False
    if intro_n == desc_n:
        return True
    if intro_n in desc_n or desc_n in intro_n:
        return True
    ratio = SequenceMatcher(None, intro_n, desc_n).ratio()
    if ratio >= 0.72:
        return True
    intro_words = set(intro_n.split())
    desc_words = set(desc_n.split())
    if intro_words and len(intro_words) >= 8:
        overlap = len(intro_words & desc_words) / max(1, len(intro_words))
        if overlap >= 0.82:
            return True
    return False

# Build one batch prompt so Gemini can write introductions for multiple products together.
def build_introduction_prompt(user_query: str, products: list[dict]) -> str:
    product_blocks = []
    for i, product in enumerate(products, start=1):
        product_blocks.append(
            f"{i}. {str(product.get('name', '')).strip()}\n"
            f"   - ASIN: {str(product.get('asin', '')).strip()}\n"
            f"   - Price: {str(product.get('price', '')).strip()}\n"
            f"   - Catalog Description: {str(product.get('catalog_description', '')).strip()}"
        )
    return f"""
You are a professional beauty e-commerce copywriter.

Write one short, polished, user-facing Introduction paragraph for each product below.
The Introductions will be shown on a recommendation webpage, so they must feel distinct from one another and sound like real recommendation copy.

[User Current Requirement]
{user_query}

[Products]
{chr(10).join(product_blocks)}

[Important Constraints]
- Keep the product identity exactly the same.
- Do not change or invent ASIN, title, or price.
- For EACH product, write a fresh sales-style introduction grounded in that product's own catalog description.
- Absolutely do NOT copy, quote, reorder, or lightly rewrite the catalog description.
- Each product must emphasize different appealing points if the products are different.
- Avoid repeating the same sentence pattern, opening phrase, adjectives, or recommendation logic across products.
- Mention why the product is appealing for the user's current need, but do not invent unsupported claims.
- Each Introduction should be exactly 2 full sentences.
- The tone should be warm, specific, persuasive, and product-aware.
- Avoid generic wording that could fit any product.
- Output in English only.
- Return valid JSON only.

[Required JSON Format]
{{
  "products": [
    {{"asin": "ASIN1", "introduction": "..."}},
    {{"asin": "ASIN2", "introduction": "..."}},
    {{"asin": "ASIN3", "introduction": "..."}}
  ]
}}
""".strip()

# Fallback prompt for per-product regeneration when batch output is missing or weak.
def _single_product_prompt(user_query: str, product: dict, used_openings: list[str] | None = None) -> str:
    used_openings = used_openings or []
    banned = "\n".join(f"- {x}" for x in used_openings if x)
    return f"""
You are a beauty e-commerce copywriter.

Write exactly 2 complete sentences of sales-style recommendation copy for this single product.

[User Need]
{str(user_query or '').strip()}

[Product]
Title: {str(product.get('name', '')).strip()}
ASIN: {str(product.get('asin', '')).strip()}
Price: {str(product.get('price', '')).strip()}
Catalog Description: {str(product.get('catalog_description', '')).strip()}

[Rules]
- Base the writing on the catalog description, but do NOT copy any sentence or phrase directly.
- Do NOT paste the description back or use near-paraphrase wording.
- Sound persuasive, natural, and product-specific.
- Keep the facts grounded in the source description.
- Use a sentence pattern that feels different from the previously written products.
- Avoid starting with any of these openings if possible:
{banned if banned else '- None'}
- Return JSON only in this format:
{{"asin": "{str(product.get('asin', '')).strip()}", "introduction": "..."}}
""".strip()

# Deterministic local fallback used when Gemini is unavailable or unusable.
def _fallback_introduction(user_query: str, product: dict) -> str:
    query = str(user_query or "the current beauty need").strip().lower()
    title = str(product.get("name", "This product")).strip() or "This product"
    asin = str(product.get("asin", "")).strip()
    variants = [
        f"{title} is a thoughtful pick for {query} because it feels more tailored than a random routine add-on. It reads like the kind of product you choose when you want a recommendation that sounds intentional and easy to trust.",
        f"For someone shopping around {query}, {title} comes across as a focused option with a clearer role in the routine. That makes it easier to recommend when you want something that feels considered rather than generic.",
        f"{title} makes a strong case for {query} if you want a product that feels purposeful from the start. Its overall profile sounds shopper-friendly and specific enough to stand out from more interchangeable options.",
    ]
    idx = sum(ord(c) for c in (asin or title)) % len(variants)
    return variants[idx]

# Extract the first JSON object from a response that may contain extra text.
def _extract_json_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0).strip() if match else ""

# Parse either JSON output or semi-structured text into an {asin: introduction} map.
def _parse_introduction_blocks(text: str) -> dict[str, str]:
    intro_map = {}
    json_text = _extract_json_text(text)
    if json_text:
        try:
            data = json.loads(json_text)
            if isinstance(data, dict) and 'asin' in data and 'introduction' in data:
                asin = str(data.get('asin', '')).strip()
                intro = _clean_text(str(data.get('introduction', ''))).strip(' -\n\t')
                if asin and intro:
                    intro_map[asin] = intro
            for item in data.get("products", []) if isinstance(data, dict) else []:
                asin = str(item.get("asin", "")).strip()
                intro = _clean_text(str(item.get("introduction", ""))).strip(" -\n\t")
                if asin and intro:
                    intro_map[asin] = intro
        except Exception:
            pass

    if intro_map:
        return intro_map

    block_pattern = re.compile(r"(?ms)^\s*(\d+)\.\s+.*?(?=\n\s*\d+\.\s+|$)")
    for block in block_pattern.finditer(text or ""):
        block_text = block.group(0).strip()
        asin_match = re.search(r"(?im)^\s*-?\s*ASIN:\s*(.+)$", block_text)
        intro_match = re.search(
            r"(?ims)^\s*-?\s*Introduction:\s*(.+?)(?=\n\s*\d+\.\s+|$)",
            block_text,
        )
        if asin_match and intro_match:
            asin = asin_match.group(1).strip()
            intro = _clean_text(intro_match.group(1)).strip(" -\n\t")
            if asin and intro:
                intro_map[asin] = intro
    return intro_map

# Use the first few normalized words to detect repetitive openings.
def _opening_signature(text: str) -> str:
    words = _normalize_for_similarity(text).split()
    return " ".join(words[:4]).strip()

# Prevent multiple recommended products from sounding almost identical.
def _is_too_similar_to_existing(intro: str, existing: list[str]) -> bool:
    intro_n = _normalize_for_similarity(intro)
    if not intro_n:
        return True
    intro_open = _opening_signature(intro)
    for other in existing:
        other_n = _normalize_for_similarity(other)
        if not other_n:
            continue
        if _opening_signature(other) == intro_open and intro_open:
            return True
        if SequenceMatcher(None, intro_n, other_n).ratio() >= 0.7:
            return True
    return False

# Generate introductions with a batch-first, single-item-second, fallback-last strategy.
def generate_product_introductions(user_query: str, products: list[dict]) -> dict[str, str]:
    if not products:
        return {}

    intro_map: dict[str, str] = {}
    accepted_intros: list[str] = []

    if client:
        # First try a single batch call for efficiency and consistent style.
        prompt = build_introduction_prompt(user_query=user_query, products=products)
        text = _call_gemini(prompt, max_output_tokens=1600)
        batch_map = _parse_introduction_blocks(text)

        for product in products:
            asin = str(product.get("asin", "")).strip()
            desc = str(product.get("catalog_description", "")).strip()
            intro = str(batch_map.get(asin, "")).strip()
            if intro and not _looks_too_close_to_description(intro, desc) and not _is_too_similar_to_existing(intro, accepted_intros):
                intro_map[asin] = intro
                accepted_intros.append(intro)

        # Retry missing items one-by-one so a weak batch response does not ruin the whole result.
        for product in products:
            asin = str(product.get("asin", "")).strip()
            desc = str(product.get("catalog_description", "")).strip()
            if not asin or asin in intro_map:
                continue
            single_text = _call_gemini(
                _single_product_prompt(user_query, product, used_openings=[_opening_signature(x) for x in accepted_intros]),
                max_output_tokens=260,
            )
            single_map = _parse_introduction_blocks(single_text)
            single_intro = str(single_map.get(asin, "")).strip()
            if single_intro and not _looks_too_close_to_description(single_intro, desc) and not _is_too_similar_to_existing(single_intro, accepted_intros):
                intro_map[asin] = single_intro
                accepted_intros.append(single_intro)

    # Final guarantee: every displayed product gets some introduction text.
    for product in products:
        asin = str(product.get("asin", "")).strip()
        if asin and not str(intro_map.get(asin, "")).strip():
            fallback = _fallback_introduction(user_query, product)
            if _is_too_similar_to_existing(fallback, accepted_intros):
                fallback = _fallback_introduction(f"{user_query} {asin}", product)
            intro_map[asin] = fallback
            accepted_intros.append(fallback)
    return intro_map