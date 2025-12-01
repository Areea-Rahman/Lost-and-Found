# =======================
# LOST & FOUND INTAKE SYSTEM (Chroma Vector DB)
# =======================

import json
import re
import sqlite3
import os 
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document 


# -----------------------
# BASIC CONFIG / THEME
# -----------------------

MODEL_NAME = "gemini-2.0-flash"
MOCK_IMAGE_URL = "https://placehold.co/100x100/A8A8A8/000000?text=IMG" 


# Simple custom CSS for nicer cards / headers
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #555555;
        margin-bottom: 0.8rem;
    }
    .card {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #E3E3E3;
        background-color: #FAFAFA;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        border-radius: 12px;
        padding: 0.8rem 1rem;
        border: 1px solid #E3E3E3;
        background-color: #FFFFFF;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        color: #777777;
    }
    .metric-card p {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------
# SECRETS / CLIENTS / VECTOR STORE
# -----------------------

@st.cache_resource
def get_secrets() -> Dict[str, Any]:
    """Load API keys and config from Streamlit secrets."""
    try:
        s = st.secrets.to_dict()
    except Exception:
        s = {}
    return {
        "raw": s,
        "gemini_key": s.get("GEMINI_API_KEY"),
        "openai_key": s.get("OPENAI_API_KEY"),
    }


secrets = get_secrets()


@st.cache_resource
def get_gemini_client():
    """Initialize Gemini client."""
    if not secrets["gemini_key"]:
        st.error("GEMINI_API_KEY is not set in Streamlit secrets.")
        return None
    try:
        return genai.Client(api_key=secrets["gemini_key"])
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None


gemini_client = get_gemini_client()


@st.cache_resource
def get_embeddings():
    if not secrets["openai_key"]:
        st.warning("OPENAI_API_KEY is not set; semantic matching will be disabled.")
        return None
    try:
        # Note: Chroma defaults to using the passed embedding function
        return OpenAIEmbeddings(openai_api_key=secrets["openai_key"])
    except Exception as e:
        st.error(f"Error creating OpenAI embeddings: {e}")
        return None


@st.cache_resource
def get_vector_store():
    """
    Create or load a Chroma vector DB using OpenAI embeddings.
    This stores all *found* items with metadata.
    """
    embeddings = get_embeddings()
    if embeddings is None:
        return None

    try:
        vs = Chroma(
            collection_name="lost_and_found_items",
            embedding_function=embeddings,
            persist_directory="chroma_db",  # local folder for persistence
        )
        st.success("Chroma vector store initialized.")
        return vs
    except Exception as e:
        st.error(f"Error creating Chroma vector store: {e}")
        return None


vector_store = get_vector_store()


# -----------------------
# SAFE GEMINI HELPERS
# -----------------------

def safe_send(chat, message_content: str, context: str = ""):
    """Wrapper around chat.send_message with clear error reporting."""
    try:
        return chat.send_message(message_content)
    except genai_errors.ClientError as e:
        label = context or "chat"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


def safe_generate(full_prompt: str, context: str = ""):
    """Wrapper around generate_content with clear error reporting."""
    if gemini_client is None:
        st.error("Gemini client is not available.")
        st.stop()
    try:
        return gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
        )
    except genai_errors.ClientError as e:
        label = context or "generation"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


# -----------------------
# PROMPTS
# -----------------------

GENERATOR_SYSTEM_PROMPT = """
Role:
You are a Lost & Found intake operator for a public-transit system. Your job is to gather accurate factual information
about a found item, refine the description interactively with the user, and output a single final structured record.

Behavior Rules:
1. Input Handling
The user may provide either an image or a short text description.
If an image is provided, describe visible traits such as color, material, type, size, markings, and notable features.
If text is provided, restate and cleanly summarize it in factual language.
Do not wait for confirmation before giving the first description.

2. Clarification
Ask targeted concise follow up questions to collect identifying details such as brand, condition,
writing, contents, location (station), and time found.
If the user provides a station name (for example “Times Sq”, “Queensboro Plaza”), try to identify the corresponding subway line or lines.
If multiple lines serve the station, you can mention all of them. If the station name has four or more lines, record only the station name.
If the station is unclear or unknown, set Subway Location to null.
Stop asking questions once the description is clear and specific enough.
Do not include questions or notes in the final output.

3. Finalization
When you have enough detail, output only this structured record:

Subway Location: <station or null>
Color: <dominant or user provided colors or null>
Item Category: <free text category such as Bags and Accessories, Electronics, Clothing or null>
Item Type: <free text item type such as Backpack, Phone, Jacket or null>
Description: <concise free text summary combining all verified details>
"""

USER_SIDE_GENERATOR_PROMPT = """
You are a helpful assistant for riders reporting lost items on a subway system.

Input:
The user may provide an image or a short text description of the lost item.
If an image is provided, describe what you see, including color, material, size, shape, and any markings.
If text is provided, restate the description in clean factual language.

Clarification:
Then ask two to four short follow up questions to collect details such as:
color if unclear, brand or logo, contents if it is a bag, any writing, where it was lost,
and approximate time.

When you have enough information, output only this structured record:

Subway Location: <station name or null>
Color: <color or colors or null>
Item Category: <category or null>
Item Type: <type or null>
Description: <concise factual summary>

Do not include your questions or reasoning in the final structured record.
"""

STANDARDIZER_PROMPT = """
You are the Lost and Found Data Standardizer for a public transit system.
You receive structured text from another model describing an item.
Your task is to map free text fields to standardized tag values and produce a clean JSON record.

Tag Source:
All valid standardized values are in the provided Tags Excel reference summary.
Use only those lists to choose values.

Field rules:

Subway Location:
Compare only with the Subway Location tag list.
Color:
Compare only with the Color tag list.
Item Category:
Compare only with the Item Category tag list.
Item Type:
Compare only with the Item Type tag list.

Use exact or closest textual matches from the correct list only.
If no good match exists return "null" for that field.

Input format:

Subway Location: <value or null>
Color: <value or null>
Item Category: <value or null>
Item Type: <value or null>
Description: <free text description>

Output:

Return only a JSON object of this form:

{
  "subway_location": ["<line or station>", "<line or station>"],
  "color": ["<color1>", "<color2>"],
  "item_category": "<standardized category or null>",
  "item_type": ["<type1>", "<type2>"],
  "description": "<clean description>",
  "time": "<ISO 8601 UTC timestamp>"
}

If a field has a single value it is still an array where the specification says array.
If you cannot confidently match a value, use "null" or an empty array as appropriate.

Do not output any explanation. Only output the JSON object.
"""


# -----------------------
# TAG / DATA HELPERS
# -----------------------

@st.cache_data
def load_tag_data():
    """Load Tags.xlsx and prepare tag lists."""
    try:
        # Create a dummy Tags.xlsx if it doesn't exist for running the code
        if not os.path.exists("Tags.xlsx"):
            mock_data = {
                "Subway Location": ["Times Square", "Union Square", "Grand Central", "null"],
                "Color": ["Red", "Blue", "Black", "Brown", "Metallic", "null"],
                "Item Category": ["Electronics", "Bags", "Clothing", "Accessories", "null"],
                "Item Type": ["Phone", "Backpack", "Wallet", "Jacket", "Keys", "null"],
            }
            pd.DataFrame(mock_data).to_excel("Tags.xlsx", index=False)
            st.info("Created mock 'Tags.xlsx' file for demonstration.")
            
        df = pd.read_excel("Tags.xlsx")
        return {
            "df": df,
            "locations": sorted(set(df["Subway Location"].dropna().astype(str))),
            "colors": sorted(set(df["Color"].dropna().astype(str))),
            "categories": sorted(set(df["Item Category"].dropna().astype(str))),
            "item_types": sorted(set(df["Item Type"].dropna().astype(str))),
        }
    except Exception as e:
        st.error(f"Error loading tag data (Tags.xlsx): {e}")
        return None


def extract_field(text_block: str, field: str) -> str:
    """Extract 'Field: value' from a structured block."""
    match = re.search(rf"{field}:\s*(.*)", text_block)
    return match.group(1).strip() if match else "null"


def is_structured_record(message: str) -> bool:
    """Detect if a message is the final structured record."""
    return message.strip().startswith("Subway Location:")


def standardize_description(text_block: str, tags: Dict) -> Dict:
    """Send structured text + tag summary to Gemini and parse JSON."""
    if gemini_client is None:
        st.error("Gemini client is not available. Cannot standardize description.")
        return {}

    tags_summary = (
        "\n--- TAGS REFERENCE ---\n"
        f"Subway Location tags: {', '.join(tags['locations'][:50])}\n"
        f"Color tags: {', '.join(tags['colors'][:50])}\n"
        f"Item Category tags: {', '.join(tags['categories'][:50])}\n"
        f"Item Type tags: {', '.join(tags['item_types'][:50])}\n"
    )

    full_prompt = f"{STANDARDIZER_PROMPT}\n\nHere is the structured input to standardize:\n{text_block}\n{tags_summary}"

    # Use safe_generate so we see real errors
    response = safe_generate(full_prompt, context="standardize_description")

    try:
        cleaned = response.text.strip()
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        json_text = cleaned[json_start:json_end]
        data = json.loads(json_text)

        # Fill missing time
        if "time" not in data or not data["time"]:
            data["time"] = datetime.now(timezone.utc).isoformat()

        # Normalize lists
        for key in ["subway_location", "color", "item_type"]:
            if key in data and isinstance(data[key], str):
                data[key] = [data[key]]
            elif key not in data or data[key] is None:
                data[key] = []
            elif key in data and data[key] == ["null"]:
                data[key] = []

        if "item_category" not in data or data["item_category"] is None:
            data["item_category"] = "null"

        if "description" not in data:
            data["description"] = extract_field(text_block, "Description")

        return data

    except Exception:
        st.error("Model output could not be parsed as JSON. Raw output below:")
        st.text(response.text)
        return {}


# -----------------------
# SIMPLE VALIDATION HELPERS
# -----------------------

def validate_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", phone))


def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


# -----------------------
# VECTOR STORE HELPERS
# -----------------------

def get_all_found_items_raw() -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Return raw ids, documents, metadatas from Chroma."""
    if vector_store is None:
        return [], [], []
    try:
        coll = vector_store._collection
        data = coll.get()
        # Data is returned as dict with keys 'ids', 'documents', 'metadatas'
        return data.get("ids", []), data.get("documents", []), data.get("metadatas", [])
    except Exception:
        return [], [], []


def get_all_found_items_as_df() -> pd.DataFrame:
    """Pull all 'found' items from Chroma for admin view & metrics."""
    ids, docs, metas = get_all_found_items_raw()
    rows: List[Dict[str, Any]] = []

    for id_, doc, meta in zip(ids, docs, metas):
        if not meta:
            continue
        if meta.get("record_type") != "found":
            continue

        rows.append(
            {
                "found_id": meta.get("found_id", id_),
                "description": doc,  # Use the document content for full description
                "subway_location": meta.get("subway_location", ""),
                "color": meta.get("color", ""),
                "item_category": meta.get("item_category", ""),
                "item_type": meta.get("item_type", ""),
                "contact": meta.get("contact", ""),
                "time": meta.get("time", ""),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def get_next_found_id() -> int:
    """Generate a simple incremental ID for found items (for display/admin)."""
    if "next_found_id" not in st.session_state:
        df = get_all_found_items_as_df()
        if df.empty:
            st.session_state.next_found_id = 1
        else:
            # Safely cast and increment the maximum ID
            try:
                max_id = df["found_id"].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else 0).max()
                st.session_state.next_found_id = max_id + 1
            except:
                st.session_state.next_found_id = 1
    nid = st.session_state.next_found_id
    st.session_state.next_found_id += 1
    return nid


def save_found_item_to_vectorstore(json_data: Dict, contact: str) -> int:
    """
    Add a found item into Chroma vector DB with metadata.
    FIXED: Converts list metadata to strings for vector store compatibility.
    """
    if vector_store is None:
        st.error("Vector store is not available; cannot save found item.")
        return -1

    description = json_data.get("description", "")
    if not description:
        st.error("Found item description is empty; cannot embed.")
        return -1

    found_id = get_next_found_id()

    # 1. Prepare Metadata (Convert all list fields to comma-separated strings)
    metadata = {
        "record_type": "found",
        "found_id": str(found_id),
        "image_path": MOCK_IMAGE_URL, # Using mock URL for now
        
        # --- FIX APPLIED HERE: CONVERTING LISTS TO STRINGS ---
        "subway_location": ", ".join(json_data.get("subway_location", [])),
        "color": ", ".join(json_data.get("color", [])),
        "item_category": json_data.get("item_category", "null"),
        "item_type": ", ".join(json_data.get("item_type", [])),
        # ----------------------------------------------------
        
        "description": description,
        "contact": contact,
        "time": json_data.get("time"),
    }

    try:
        # 2. Store the document and its simple metadata in Chroma
        vector_store.add_texts(
            texts=[description],
            metadatas=[metadata],
            ids=[str(found_id)],
        )
        # Note: Chroma auto-persists in the defined directory
        st.success(f"Found item saved with ID: {found_id} and embedded in Chroma.")
        return found_id
    except Exception as e:
        # Catch and log the error specifically related to vector storage
        st.error(f"Error saving found item to vector store: {e}")
        return -1


def search_matches_for_lost_item(
    final_json: Dict, top_k: int, max_distance: float
) -> Tuple[List[Any], List[Any]]:
    """
    Performs Hybrid Search: Metadata Filter (Category) + Vector Search (Cosine).
    """
    if vector_store is None:
        st.warning("Vector store is not available for matching. Returning mock results.")
        # Mock result for visual display if Chroma is down
        return (
            [
                (Document(page_content="Mock Red Backpack", metadata={"found_id": "M99", "item_category": "Bags", "color": "Red"}), 0.95),
                (Document(page_content="Mock Black Phone", metadata={"found_id": "M98", "item_category": "Electronics", "color": "Black"}), 0.88)
            ],
            [
                (Document(page_content="Mock Red Backpack", metadata={"found_id": "M99", "item_category": "Bags", "color": "Red"}), 0.95),
            ]
        )

    # 1. Extract Structured Filters
    category_filter = final_json.get("item_category")
    
    langchain_filter: Dict[str, str] = {}
    
    # Only apply category filter if valid
    if category_filter and category_filter != "null":
        # Chroma/LangChain filtering syntax (exact match)
        langchain_filter["item_category"] = category_filter

    query_text = final_json.get("description", "")
    if not query_text:
        return [], []
    
    st.info(f"Hybrid Search Query: '{query_text[:30]}...' | Filter: {langchain_filter}")

    # 2. Execute Hybrid Search
    # similarity_search_with_score returns a list of tuples: (Document, score)
    try:
        docs_scores = vector_store.similarity_search_with_score(
            query_text,
            k=top_k,
            where=langchain_filter # Use 'where' for metadata filtering in Chroma
        )
    except Exception as e:
        st.error(f"Error during vector search query: {e}")
        docs_scores = []

    # 3. Filter by distance threshold (post-processing)
    # Chroma returns distance (lower score = closer/more similar)
    all_candidates = docs_scores
    filtered = [(doc, score) for doc, score in docs_scores if score <= max_distance]
    
    return all_candidates, filtered


# -----------------------
# APP INIT
# -----------------------

st.set_page_config(
    page_title="Lost & Found Intake",
    page_icon="🧳",
    layout="wide",
)

# Ensure the mock file exists
if not os.path.exists("Tags.xlsx"):
    load_tag_data()

# Ensure database tables exist
# (SQLite is only used for metadata persistence outside of the vector store here)
with sqlite3.connect("lost_and_found.db") as conn:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lost_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT,
            contact TEXT,
            email TEXT,
            json_data TEXT
        )
    """
    )


# -----------------------
# SIDEBAR NAV + MATCHING CONTROLS
# -----------------------

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["👮 Operator: Upload Found Item", "🧍 User: Report Lost Item", "📊 Admin: View Found Items"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Matching Controls")

top_k_sidebar = st.sidebar.slider(
    "Top-K candidates",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Number of candidate matches to retrieve from the vector DB.",
)

max_distance_sidebar = st.sidebar.slider(
    "Max distance (lower = more similar)",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.01,
    help="Matches with distance greater than this will be filtered out.",
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Distance is model-dependent. Start with 0.4–0.5 and adjust.")


# -----------------------
# TOP DASHBOARD METRICS
# -----------------------

df_found_for_metrics = get_all_found_items_as_df()
total_found = len(df_found_for_metrics)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown('<div class="metric-card"><h3>Total Found Items</h3><p>{}</p></div>'.format(total_found), unsafe_allow_html=True)
with col_m2:
    st.markdown(
        '<div class="metric-card"><h3>Matching Top-K</h3><p>{}</p></div>'.format(top_k_sidebar),
        unsafe_allow_html=True,
    )

st.markdown("")  # spacing


# ===============================================================
# PAGE 1: OPERATOR – UPLOAD FOUND ITEM
# ===============================================================

if page.startswith("👮"):
    st.markdown('<div class="main-title">👮 Operator View: Upload Found Item</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Frontline staff can quickly record what they found, '
        'then the system standardizes tags and stores it in the vector database for matching later.</div>',
        unsafe_allow_html=True,
    )

    if "operator_chat" not in st.session_state:
        st.session_state.operator_chat = gemini_client.chats.create(
            model= MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM_PROMPT,
            ),
        )
        st.session_state.operator_msgs = []

    # Show conversation history
    if st.session_state.operator_msgs:
        st.markdown('<div class="section-title">🗨️ Intake Conversation</div>', unsafe_allow_html=True)
    for msg in st.session_state.operator_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Initial intake
    if not st.session_state.operator_msgs:
        st.markdown('<div class="section-title">➕ Start a New Found Item</div>', unsafe_allow_html=True)
        with st.container():
            col1, col2 = st.columns(2)
            uploaded_image = None
            initial_text = ""
            
            with col1:
                uploaded_image = st.file_uploader(
                    "📷 Image of the found item (optional)",
                    type=["jpg", "jpeg", "png"],
                    key="operator_image",
                )
                if uploaded_image:
                     st.image(Image.open(uploaded_image).convert("RGB"), width=220, caption="Preview of found item")

            with col2:
                initial_text = st.text_input(
                    "📝 Short description",
                    placeholder="For example: black backpack with a NASA patch",
                    key="operator_text",
                )

            if st.button("🚀 Start Intake"):
                if not uploaded_image and not initial_text:
                    st.error("Please upload an image or enter a short description.")
                else:
                    message_content = ""
                    if uploaded_image:
                        message_content += "I have uploaded an image of the found item. "
                    if initial_text:
                        message_content += initial_text

                    st.session_state.operator_msgs.append(
                        {"role": "user", "content": message_content}
                    )
                    with st.spinner("Analyzing item with Gemini..."):
                        response = safe_send(
                            st.session_state.operator_chat,
                            message_content,
                            context="operator intake",
                        )
                    st.session_state.operator_msgs.append(
                        {"role": "model", "content": response.text}
                    )
                    st.rerun()

    # Continue chat
    operator_input = st.chat_input("Add more details for the operator bot, or say 'done' when ready.")
    if operator_input:
        st.session_state.operator_msgs.append(
            {"role": "user", "content": operator_input}
        )
        with st.spinner("Processing operator message..."):
            response = safe_send(
                st.session_state.operator_chat,
                operator_input,
                context="operator follow-up",
            )
        st.session_state.operator_msgs.append(
            {"role": "model", "content": response.text}
        )
        st.session_state.operator_input = "" # Clear input
        st.rerun()


    # When final structured record appears
    if st.session_state.operator_msgs and is_structured_record(
        st.session_state.operator_msgs[-1]["content"]
    ):
        structured_text = st.session_state.operator_msgs[-1]["content"]
        st.markdown('<div class="section-title">📦 Final Structured Description</div>', unsafe_allow_html=True)
        st.code(structured_text)

        final_json = standardize_description(structured_text, tag_data)
        if final_json:
            st.markdown('<div class="section-title">🏷️ Standardized Tags (JSON)</div>', unsafe_allow_html=True)
            st.json(final_json)

            st.markdown('<div class="section-title">👤 Operator Contact</div>', unsafe_allow_html=True)
            contact = st.text_input("Operator contact or badge ID")

            if st.button("💾 Save Found Item to Vector DB"):
                found_id = save_found_item_to_vectorstore(final_json, contact)
                if found_id > 0:
                    st.success(f"Found item saved with ID `{found_id}` (Chroma vector DB).")
                    st.session_state.operator_msgs = [] # Clear chat on success
                    st.rerun()


# ===============================================================
# PAGE 2: USER – REPORT LOST ITEM & MATCH
# ===============================================================

if page.startswith("🧍"):
    st.markdown('<div class="main-title">🧍 User View: Report Lost Item</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Riders describe what they lost. The system standardizes their description '
        'and searches for similar found items using embeddings.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">⚡ Optional Quick Info</div>', unsafe_allow_html=True)
    with st.expander("Click to pre-select station / category / type"):
        col1, col2, col3 = st.columns(3)
        with col1:
            location_choice = st.selectbox(
                "🚉 Subway station (optional)", [""] + tag_data["locations"]
            )
        with col2:
            category_choice = st.selectbox(
                "📂 Item category (optional)", [""] + tag_data["categories"]
            )
        with col3:
            type_choice = st.selectbox(
                "🔖 Item type (optional)", [""] + tag_data["item_types"]
            )

    st.markdown('<div class="section-title">📷 / 📝 Describe Your Lost Item</div>', unsafe_allow_html=True)
    col_img, col_text = st.columns(2)
    with col_img:
        uploaded_image = st.file_uploader(
            "Image of lost item (optional)",
            type=["jpg", "jpeg", "png"],
            key="user_image",
        )
    with col_text:
        initial_text = st.text_input(
            "Short description",
            placeholder="For example: blue iPhone with cracked screen",
            key="user_text",
        )

    if "user_chat" not in st.session_state:
        st.session_state.user_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=USER_SIDE_GENERATOR_PROMPT,
            ),
        )
        st.session_state.user_msgs = []

    # Show chat history
    if st.session_state.user_msgs:
        st.markdown('<div class="section-title">🗨️ Chat with the Lost-Item Assistant</div>', unsafe_allow_html=True)
    for msg in st.session_state.user_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Start report
    if not st.session_state.user_msgs and st.button("🚀 Start Lost Item Report"):
        if not uploaded_image and not initial_text:
            st.error("Please upload an image or enter a short description.")
        else:
            message_text = ""
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, width=240, caption="Your lost item (preview)")
                message_text += "I have uploaded an image of my lost item. "
            if initial_text:
                message_text += initial_text

            st.session_state.user_msgs.append(
                {"role": "user", "content": message_text}
            )
            with st.spinner("Analyzing your description..."):
                response = safe_send(
                    st.session_state.user_chat,
                    message_text,
                    context="user initial report",
                )
            st.session_state.user_msgs.append(
                {"role": "model", "content": response.text}
            )
            st.rerun()

    # Continue chat
    user_input = st.chat_input("Add more details, answer questions, or say 'done' when ready.")
    if user_input:
        st.session_state.user_msgs.append(
            {"role": "user", "content": user_input}
        )
        with st.spinner("Thinking..."):
            response = safe_send(
                st.session_state.user_chat,
                user_input,
                context="user follow-up",
            )
        st.session_state.user_msgs.append(
            {"role": "model", "content": response.text}
        )
        st.rerun()

    # When final structured record appears
    if st.session_state.user_msgs and is_structured_record(
        st.session_state.user_msgs[-1]["content"]
    ):
        structured_text = st.session_state.user_msgs[-1]["content"]

        # 1. Merge dropdown choices with LLM output
        merged_text = f"""
Subway Location: {location_choice or extract_field(structured_text, 'Subway Location')}
Color: {extract_field(structured_text, 'Color')}
Item Category: {category_choice or extract_field(structured_text, 'Item Category')}
Item Type: {type_choice or extract_field(structured_text, 'Item Type')}
Description: {extract_field(structured_text, 'Description')}
        """

        st.markdown('<div class="section-title">1. Standardizing Record</div>', unsafe_allow_html=True)
        final_json = standardize_description(merged_text, tag_data)
        if final_json:
            st.success("Standardized record created.")
            st.json(final_json)

            # 2. MATCHING WORKSTREAM (Hybrid Search)
            st.markdown('<div class="section-title">2. Searching for Found Items (Hybrid Match)</div>', unsafe_allow_html=True)
            
            if vector_store is None:
                st.error("Cannot perform matching: Vector store not configured.")
            else:
                with st.spinner("Executing Hybrid Vector Search..."):
                    all_candidates, filtered = search_matches_for_lost_item(
                        final_json,
                        top_k=top_k_sidebar,
                        max_distance=max_distance_sidebar,
                    )
                
                if not all_candidates:
                    st.info("No items are stored in the vector DB yet, so no matches can be returned.")
                else:
                    st.markdown('<div class="section-title">📌 Candidate Matches</div>', unsafe_allow_html=True)

                    if not filtered:
                        st.info(
                            "No matches under the current distance threshold. "
                            "Showing raw top-K candidates instead."
                        )
                        to_show = all_candidates
                    else:
                        to_show = filtered

                    for doc, score in to_show:
                        meta = doc.metadata or {}
                        # Cosine distance is used; score close to 0 is high similarity (1.0 is max distance)
                        similarity_pct = max(0.0, (1.0 - score) * 100.0)

                        st.markdown(
                            f"**Distance:** `{score:.4f}` · "
                            f"**Similarity (approx):** `{similarity_pct:.1f}%`"
                        )

                        col_details, col_image = st.columns([4, 1])
                        with col_details:
                            st.write("**Description:**", meta.get("description", doc.page_content))
                            st.caption(f"Category: {meta.get('item_category', 'N/A')}, Color: {meta.get('color', 'N/A')}")
                            st.caption(f"Found item ID: {meta.get('found_id', 'N/A')}")
                        with col_image:
                            st.image(meta.get("image_path", MOCK_IMAGE_URL), caption="Found Item", width=100)

                        st.markdown("---")


            # 3. CONTACT AND SUBMISSION
            st.markdown('<div class="section-title">3. Contact information</div>', unsafe_allow_html=True)
            contact = st.text_input("Phone number, ten digits")
            email = st.text_input("Email address")

            if st.button("Submit Lost Item Report"):
                if not validate_phone(contact):
                    st.error("Please enter a ten digit phone number without spaces.")
                elif not validate_email(email):
                    st.error("Please enter a valid email address.")
                else:
                    add_lost_item(
                        final_json.get("description", ""),
                        contact,
                        email,
                        json.dumps(final_json),
                    )
                    st.success("Lost item report submitted and potential matches identified.")
                    st.session_state.user_msgs = []
                    st.rerun()


# ===============================================================
# PAGE 3: ADMIN – VIEW FOUND ITEMS
# ===============================================================

if page.startswith("📊"):
    st.markdown('<div class="main-title">📊 Admin: View Stored Found Items</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Quick view of everything operators have logged into the vector DB.</div>',
        unsafe_allow_html=True,
    )

    if vector_store is None:
        st.error("Vector store is not available.")
    else:
        df_found = get_all_found_items_as_df()
        if df_found.empty:
            st.info("No found items stored yet.")
        else:
            st.markdown('<div class="section-title">📦 Found Items Table</div>', unsafe_allow_html=True)
            st.dataframe(df_found, use_container_width=True)
            st.caption("Scroll horizontally to see all columns.")
