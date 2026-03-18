"""Product recommendation page."""
import os
import re
import time
import threading
from io import BytesIO
import pandas as pd
import requests
import streamlit as st
from main_app import (
    load_table2,
    load_vector,
    build_product_descriptions_from_asins,
    get_shortlisted_products,
    build_shortlisted_catalog_context_from_products,
)
from llm_assistant import generate_recommendation, generate_product_introductions

# Configure the Streamlit page before any visible UI is rendered.
st.set_page_config(
    page_title="Intelligent Beauty Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS so the page looks more like a product recommendation site.
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2d1f3d;
        margin-bottom: 0.35rem;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: #5c4d6d;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .section-card {
        background: #ffffff;
        border: 1px solid #ececec;
        border-radius: 14px;
        padding: 0.55rem 0.9rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        margin-bottom: 0.75rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d1f3d;
        margin: 0;
        line-height: 1.2;
    }

    .result-card {
        background: #fcfcff;
        border: 1px solid #e8e8f3;
        border-radius: 14px;
        padding: 0.55rem 0.9rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        margin-top: 0.8rem;
        margin-bottom: 0.75rem;
    }

    .product-card {
        background: #ffffff;
        border: 1px solid #ececf5;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    
    .summary-box {
        background: #fff9fc;
        border-left: 4px solid #d9b3c9;
        padding: 0.85rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 2.6rem;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 1.7rem;
    }

    .footer-note {
        text-align: center;
        color: #7a7a7a;
        font-size: 0.9rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

</style>
""", unsafe_allow_html=True)

# Cache CSV loading so reruns do not keep re-reading large files from disk.
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    table2_path = os.path.join(base_dir, "table2_metadata_merged.csv")
    vector_path = os.path.join(base_dir, "vector_llm_tasks.csv")

    table2_df = load_table2(table2_path)
    vector_df = load_vector(vector_path)
    return table2_df, vector_df

""" Auto-exit protection: if the browser is closed and no more reruns happen, 
    this watchdog can terminate the Python process after a period of inactivity."""
INACTIVITY_EXIT_SECONDS = int(os.getenv("STREAMLIT_INACTIVITY_EXIT_SECONDS", "180"))
_LAST_ACTIVITY_TS = time.time()
_WATCHDOG_STARTED = False

# Refresh the last-seen activity time whenever the page is used.
def touch_activity_timestamp():
    global _LAST_ACTIVITY_TS
    _LAST_ACTIVITY_TS = time.time()

# Background thread that exits the process after long inactivity.
def _inactivity_watchdog():
    while True:
        time.sleep(5)
        if time.time() - _LAST_ACTIVITY_TS > INACTIVITY_EXIT_SECONDS:
            os._exit(0)

# Start the watchdog only once to avoid duplicate background threads.
def ensure_inactivity_watchdog_started():
    global _WATCHDOG_STARTED
    if _WATCHDOG_STARTED:
        return
    thread = threading.Thread(target=_inactivity_watchdog, daemon=True)
    thread.start()
    _WATCHDOG_STARTED = True

# Resolve either a row number or a user_id into one vector-data row.
def get_user_row(vector_df: pd.DataFrame, user_input: str):
    user_input = str(user_input).strip()

    if not user_input:
        return vector_df.iloc[0], ""

    if user_input.isdigit():
        idx = int(user_input)
        if idx < 0 or idx >= len(vector_df):
            return vector_df.iloc[0], f"Row number is out of range. Please enter a value between 0 and {len(vector_df) - 1}."
        return vector_df.iloc[idx], ""

    matched = vector_df[vector_df["user_id"].astype(str) == user_input]
    if matched.empty:
        return vector_df.iloc[0], "The user ID was not found. The default user (row 0) has been used."
    return matched.iloc[0], ""

""" Parse Gemini plain-text output into:
    - summary
    - products: [{rank, name, asin, price, introduction}]
    - final_suggestion
    Accept either Introduction or Description from the model."""
def parse_gemini_output(result_text: str):
    summary = ""
    final_suggestion = ""
    products = []
    text = (result_text or "").strip()

    summary_match = re.search(
        r"Recommendation Summary:\s*(.*?)(?:Recommended Products:|$)",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    if summary_match:
        summary = summary_match.group(1).strip()

    final_match = re.search(
        r"Final Suggestion:\s*(.*?)(?:If the user has any other needs.*|$)",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    if final_match:
        final_suggestion = final_match.group(1).strip()

    # Split the recommendation body into numbered product blocks.
    block_pattern = re.compile(
        r"(?ms)^\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.\s+|\n\s*Final Suggestion:|$)"
    )

    for match in block_pattern.finditer(text):
        rank = match.group(1).strip()
        block_text = match.group(0).strip()

        name_match = re.search(r"^\s*\d+\.\s*(.+)$", block_text, flags=re.MULTILINE)
        asin_match = re.search(r"(?im)^\s*-?\s*ASIN:\s*(.+)$", block_text)
        price_match = re.search(r"(?im)^\s*-?\s*Price:\s*(.+)$", block_text)
        intro_match = re.search(r"(?ims)^\s*-?\s*(?:Introduction|Description):\s*(.+)$", block_text)

        name = name_match.group(1).strip() if name_match else ""
        asin = asin_match.group(1).strip() if asin_match else ""
        price = price_match.group(1).strip() if price_match else ""
        introduction = intro_match.group(1).strip() if intro_match else ""

        if asin:
            products.append({
                "rank": rank,
                "name": name,
                "asin": asin,
                "price": price,
                "introduction": introduction,
            })
    return summary, products, final_suggestion

# Normalize ASINs before comparing model output with catalog rows.
def normalize_asin(asin: str) -> str:
    asin = str(asin).strip().upper()
    asin = re.sub(r"[^A-Z0-9]", "", asin)
    return asin

# If structured parsing fails, still try to recover a summary-like string.
def extract_summary_fallback(result_text: str) -> str:
    text = str(result_text or "").strip()
    if not text:
        return ""

    match = re.search(r"Recommendation Summary:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return re.sub(r"\s+", " ", text).strip()

# Look up one product row in table2 using a normalized ASIN match.
def get_product_info_from_table2_by_asin(table2_df: pd.DataFrame, asin: str):
    if not asin:
        return None

    asin_clean = normalize_asin(asin)
    asin_series = table2_df["asin"].astype(str).apply(normalize_asin)
    matched = table2_df[asin_series == asin_clean]
    if not matched.empty:
        return matched.iloc[0]
    return None

# Parse one or more image URLs from a catalog field with inconsistent formatting.
def extract_image_url_candidates(raw_image_url: str):
    raw_image_url = str(raw_image_url or "").strip()
    if raw_image_url.lower() in ["", "nan", "none"]:
        return []

    cleaned = raw_image_url.strip().strip("[]")
    cleaned = cleaned.replace("\/", "/").replace("\u0026", "&")
    cleaned = cleaned.replace("'", "").replace('\"', '"').replace('"', "")

    candidates = re.findall(r"https?://[^\s,;\]\)]+", cleaned)
    if not candidates:
        parts = [p.strip() for p in re.split(r"[,;|]", cleaned) if p.strip()]
        candidates = [p for p in parts if p.startswith("http")]

    seen = set()
    ordered = []
    for url in candidates:
        url = url.strip()
        if not url or url in seen:
            continue
        seen.add(url)
        ordered.append(url)
    return ordered

# Download image bytes for preview. Cached to avoid repeated requests on rerun.
@st.cache_data(show_spinner=False)
def fetch_image_bytes(image_url: str):
    image_url = str(image_url or "").strip()
    if not image_url:
        return None
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": "https://www.amazon.com/",
        }
        response = requests.get(image_url, headers=headers, timeout=15, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        content = response.content
        if not content:
            return None
        if "image" not in content_type.lower() and not image_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
            return None
        return content
    except Exception:
        return None

# Display helper for price text coming from table2.
def format_price_for_display(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    return text

# Rebuild product identity fields from table2.
def prepare_products_for_display(products, table2_df):
    prepared = []
    used = set()
    for product in products:
        asin_norm = normalize_asin(product.get("asin", ""))
        if not asin_norm or asin_norm in used:
            continue

        matched_row = get_product_info_from_table2_by_asin(table2_df, asin_norm)
        if matched_row is None:
            continue

        used.add(asin_norm)
        prepared.append({
            "rank": str(len(prepared) + 1),
            "name": str(matched_row.get("title", "") or "").strip(),
            "asin": str(matched_row.get("asin", "") or "").strip(),
            "price": format_price_for_display(matched_row.get("price", "")),
            "catalog_description": str(matched_row.get("description", "") or "").strip(),
            "introduction": str(product.get("introduction", "") or "").strip(),
            "image_urls": extract_image_url_candidates(matched_row.get("image_url", "")),
        })

        # Only display the top three products on the page.
        if len(prepared) >= 3:
            break
    return prepared

# Regenerate polished product introductions specifically for the display cards.
def enrich_introductions_with_gemini(user_query: str, prepared_products: list[dict]):
    if not prepared_products:
        return prepared_products

    intro_map = generate_product_introductions(user_query=user_query, products=prepared_products)
    intro_map_norm = {normalize_asin(k): v for k, v in intro_map.items() if k and v}
    for product in prepared_products:
        asin_norm = normalize_asin(product.get("asin", ""))
        new_intro = str(intro_map_norm.get(asin_norm, "") or "").strip()
        if new_intro:
            product["introduction"] = new_intro
    return prepared_products


# Start inactivity watchdog.
ensure_inactivity_watchdog_started()
touch_activity_timestamp()

# Header.
st.markdown("""
<div style="margin-bottom: 1rem;">
    <div class="hero-title">🛍️ Intelligent Beauty Product Recommendation System</div>
    <div class="hero-subtitle">
        This system combines user purchase history, predicted product interests, and real-time shopping needs
        to generate personalized beauty product recommendations through LLM.
    </div>
</div>
""", unsafe_allow_html=True)

# Load data.
table2_df, vector_df = load_data()
st.markdown("""
<div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.6rem;">User Input</div>
""", unsafe_allow_html=True)
with st.container(border=True):
    col1, col2, col3 = st.columns([1.4, 2.4, 0.9])

    with col1:
        user_input = st.text_input(
            "User row number or user ID",
            placeholder="e.g. 53 or a AHUIIYUMY5OXQWRECL5JEK4AITBA"
        )
        st.caption(f"Enter either a row number from 0 to {len(vector_df) - 1}, or a specific user ID.")
    with col2:
        user_query = st.text_input(
            "User's current requirement",
            placeholder="e.g. Recommend a cheap and effective facial cleanser."
        )
        st.caption("Describe the user's real-time shopping need in simple words.")
    with col3:
        run_button = st.button("Generate Recommendation", type="primary")

# Main action
if run_button:
    try:
        with st.spinner("Analyzing user profile and generating personalized recommendation..."):
            user_row, user_message = get_user_row(vector_df, user_input)
            if user_message:
                st.warning(user_message)

            user_id = str(user_row.get("user_id", "") or "")

            # Convert history / prediction ASIN lists into text context for the recommendation pipeline.
            history_asin_text = user_row.get("history_asin_list", None)
            history_description_text = build_product_descriptions_from_asins(
                history_asin_text,
                table2_df,
                max_items=None
            )

            pred_asin_text = user_row.get("pred_asin", None)
            predicted_description_text = build_product_descriptions_from_asins(
                pred_asin_text,
                table2_df,
                max_items=None
            )

            final_query = user_query.strip() if user_query.strip() else "Please recommend suitable beauty products for me."

            # Build a local shortlist first, then pass only a compact catalog slice downstream.
            shortlisted_products = get_shortlisted_products(
                table2_df=table2_df,
                user_query=final_query,
                historical_context=history_description_text,
                predicted_context=predicted_description_text,
                shortlist_size=8
            )
            shortlisted_catalog_context = build_shortlisted_catalog_context_from_products(
                shortlisted_products
            )

            result = generate_recommendation(
                user_id=user_id,
                user_query=final_query,
                historical_context=history_description_text,
                predicted_context=predicted_description_text,
                shortlisted_catalog_context=shortlisted_catalog_context,
                use_gemini=True
            )
            summary, products, final_suggestion = parse_gemini_output(result)
            if not str(summary or "").strip():
                summary = extract_summary_fallback(result)
            products = prepare_products_for_display(products, table2_df)

            # If the model did not return enough structured products, backfill from the shortlist.
            if len(products) < 3:
                fallback_products = []
                for item in shortlisted_products:
                    fallback_products.append({
                        "asin": item["asin"],
                        "introduction": "",
                    })
                products = prepare_products_for_display(products + fallback_products, table2_df)

            # Generate card-friendly introductions tied to each selected product.
            products = enrich_introductions_with_gemini(final_query, products)

            # Store the latest results in session state for possible later UI reuse.
            st.session_state["latest_products"] = products
            st.session_state["latest_final_suggestion"] = final_suggestion

        st.success("Recommendation generated successfully.")

        # Product cards
        if products:
            st.markdown("""
            <div style="font-size: 1.5rem; font-weight: 600; margin: 0.4rem 0 0.8rem 0; color: #2d1f3d;">
                Top Recommended Products
            </div>
            """, unsafe_allow_html=True)

            for product in products:
                image_urls = product.get("image_urls", [])
                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                st.markdown(f"**#{product['rank']} Recommendation**")
                left, right = st.columns([1, 2.2], vertical_alignment="top")

                with left:
                    displayed = False
                    # Try the available image URLs until one can be rendered successfully.
                    for image_url in image_urls:
                        image_bytes = fetch_image_bytes(image_url)
                        if image_bytes:
                            try:
                                st.image(BytesIO(image_bytes), width="stretch")
                                displayed = True
                                break
                            except Exception:
                                continue
                    if not displayed and image_urls:
                        st.markdown(f"[Open image]({image_urls[0]})")
                        st.info("Image preview unavailable")
                    elif not displayed:
                        st.info("No image available")

                with right:
                    st.markdown(
                        f"<div style='font-size: 1.3rem; font-weight: 700; margin-bottom: 0.45rem; color: #303040;'>"
                        f"{product['name']}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**ASIN:** {product['asin']}")
                    st.markdown(f"**Price:** {product['price']}")
                    st.markdown(f"**Introduction:** {product['introduction']}")

                st.markdown('</div>', unsafe_allow_html=True)
                st.write("")

        # Final suggestion
        if final_suggestion:
            st.markdown("""
                <div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.6rem;">Final Suggestion</div>
                """, unsafe_allow_html=True)
            st.info(final_suggestion)

    except Exception as e:
        st.error(str(e))

# Footer.
st.markdown("""
<div class="footer-note">
    Intelligent Beauty Product Recommendation System · Streamlit Interface
</div>
""", unsafe_allow_html=True)