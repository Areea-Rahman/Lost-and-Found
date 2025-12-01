# =======================
# LOST & FOUND INTAKE SYSTEM
# =======================

import streamlit as st
import sqlite3
import json
import re
from datetime import datetime, timezone
from PIL import Image
import pandas as pd
import os 
from typing import List, Dict, Any 

from google import genai
from google.genai import types
from google.genai import errors as genai_errors  # for catching ClientError

# Use a single constant for the model name
MODEL_NAME = "gemini-2.0-flash"

# --- Hybrid Search Constants ---
# We will use this to connect the SQLite metadata to the PGVector document
COLLECTION_NAME = "lostandfound_items"
PG_METADATA_FIELDS = ["subway_location", "color", "item_category", "item_type"]

# Mock Image Placeholder URL for Streamlit
MOCK_IMAGE_URL = "https://placehold.co/100x100/A8A8A8/000000?text=IMG" 


# Optional vector-search imports (wrapped in try so app still runs if missing)
try:
    from langchain_community.vectorstores import PGVector
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document 

    HAS_VECTOR_LIBS = True
except Exception:
    HAS_VECTOR_LIBS = False


# =======================
# SECRETS / CONFIG HELPERS
# =======================

@st.cache_resource
def get_secrets():
    """Read secrets and expose simple flags."""
    try:
        secrets_dict = st.secrets.to_dict()
    except Exception:
        secrets_dict = {}

    # NOTE: Assuming secrets are set up for OpenAI and PGVector in Streamlit secrets
    gemini_key = secrets_dict.get("GEMINI_API_KEY")
    # For PGVector, LangChain PGVector typically uses OpenAI for embeddings
    openai_key = secrets_dict.get("OPENAI_API_KEY") 
    pg_conn_string = secrets_dict.get("PG_CONNECTION_STRING")

    return {
        "raw": secrets_dict,
        "openai_key": openai_key,
        "pg_conn_string": pg_conn_string,
        "has_gemini": gemini_key is not None,
        "has_openai": openai_key is not None,
        "has_pg": pg_conn_string is not None and HAS_VECTOR_LIBS,
    }


secrets = get_secrets()


# =======================
# GEMINI CLIENT
# =======================

@st.cache_resource
def get_gemini_client():
    if not secrets["has_gemini"]:
        st.error("GEMINI_API_KEY is not set in Streamlit secrets.")
        return None
    try:
        client = genai.Client(api_key=secrets["gemini_key"])
        return client
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None


gemini_client = get_gemini_client()


# =======================
# SAFE GEMINI HELPERS
# =======================

def safe_send(chat, message_content: str, context: str = ""):
    """
    Wrapper for chat.send_message that shows the real Gemini ClientError
    instead of the redacted Streamlit traceback.
    """
    try:
        return chat.send_message(message_content)
    except genai_errors.ClientError as e:
        label = context or "chat"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


def safe_generate(full_prompt: str, context: str = ""):
    """
    Wrapper for gemini_client.models.generate_content with clear error reporting.
    """
    if gemini_client is None:
        st.error("Gemini client is not available.")
        st.stop()

    try:
        # Check if the prompt contains image data (not implemented in this mock)
        contents = full_prompt 
        
        return gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
        )
    except genai_errors.ClientError as e:
        label = context or "standardization"
        st.error(f"Gemini ClientError during {label}: {e}")
        if getattr(e, "response_json", None):
            st.json(e.response_json)
        st.stop()


# =======================
# OPTIONAL VECTOR STORE (PGVECTOR + OPENAI)
# =======================

@st.cache_resource
def get_vector_store():
    """
    Try to connect to existing PGVector store using OpenAI Embeddings.
    """
    if not secrets["has_pg"]:
        # Check happens inside get_secrets now, so this catches if the dependency install failed
        if not HAS_VECTOR_LIBS:
            st.warning("Vector search disabled: Missing LangChain/OpenAI dependencies.")
        return None

    if not (secrets["has_openai"] and secrets["has_pg"]):
        st.warning("Vector search disabled: Missing OpenAI key or PG connection string.")
        return None

    try:
        # LangChain PGVector defaults to using OpenAI Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=secrets["openai_key"])
        db = PGVector(
            connection_string=secrets["pg_conn_string"],
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            # Ensure table creation works if the collection doesn't exist
            # Note: This is a complex step in production
            pre_delete_collection=False 
        )
        
        st.success(f"Vector search enabled using collection: {COLLECTION_NAME}")
        return db
    except Exception as e:
        st.error(f"PGVector connection error: {e}. Vector search disabled.")
        return None


vector_store = get_vector_store()


# =======================
# PROMPTS (Reverted)
# =======================

GENERATOR_SYSTEM_PROMPT = """
Role:
You are a Lost & Found intake operator for a public transit system. Your job is to gather accurate factual information
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


# =======================
# DATA HELPERS
# =======================

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


def extract_field(text: str, field: str) -> str:
    """Extract a simple `Field: value` line from a structured block."""
    match = re.search(rf"{field}:\s*(.*)", text)
    return match.group(1).strip() if match else "null"


def is_structured_record(message: str) -> bool:
    """Detect if the message looks like the final structured record."""
    return message.strip().startswith("Subway Location:")


def standardize_description(text: str, tags: dict) -> dict:
    """Send structured text plus tag summary to Gemini and parse JSON."""
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

    full_prompt = f"{STANDARDIZER_PROMPT}\n\nHere is the structured input to standardize:\n{text}\n{tags_summary}"

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

        # Normalize list-type fields
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
            data["description"] = extract_field(text, "Description")

        return data
    except Exception:
        st.error("Model output could not be parsed as JSON. Raw output below:")
        st.text(response.text)
        return {}


# =======================
# DATABASE HELPERS (SQLite for metadata and Hybrid Search)
# =======================

def init_db():
    with sqlite3.connect("lost_and_found.db") as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS found_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caption TEXT,
                location TEXT,
                contact TEXT,
                image_path TEXT,
                json_data TEXT
            )
        """
        )
        c.execute(
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
        conn.commit()


def add_found_item(caption, location, contact, image_path, json_data_string):
    """
    Saves item metadata to SQLite and stores embedding in PGVector.
    """
    if vector_store is None:
        st.error("Cannot save: Vector store is not initialized.")
        return False

    with sqlite3.connect("lost_and_found.db") as conn:
        # 1. Save metadata to SQLite first
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO found_items (caption, location, contact, image_path, json_data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (caption, location, contact, image_path, json_data_string),
        )
        item_id = c.lastrowid
        conn.commit()

        # 2. Extract standardized metadata for PGVector
        json_data = json.loads(json_data_string)
        
        # PGVector requires metadata values to be simple strings/numbers
        metadata = {
            "source": "found_item",
            "sqlite_id": str(item_id), # Link back to the SQLite record
            "image_path": image_path,
            "item_category": json_data.get("item_category", "null"),
            "color": json_data["color"][0] if json_data["color"] else "null", # Use first color for filter
            "location": json_data["subway_location"][0] if json_data["subway_location"] else "null", # Use first location for filter
        }
        
        # 3. Create document and store embedding in PGVector
        # Page content is the description which will be embedded
        doc = Document(
            page_content=json_data.get("description", caption), 
            metadata=metadata
        )
        vector_store.add_documents([doc])
        st.success(f"Found item saved as SQLite ID: {item_id} and embedded in PGVector.")
        return True


def add_lost_item(description, contact, email, json_data_string):
    with sqlite3.connect("lost_and_found.db") as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO lost_items (description, contact, email, json_data)
            VALUES (?, ?, ?, ?);
            """,
            (description, contact, email, json_data_string),
        )
        conn.commit()


def hybrid_match_found_items(lost_item_json: dict, k: int = 3) -> List[Dict[str, Any]]:
    """
    Performs Hybrid Search: Metadata Filter (SQL-like) + Vector Search (Cosine).
    """
    if vector_store is None:
        # Fallback needed if vector store isn't running
        st.warning("Vector store is not available for matching. Returning mock results.")
        # Mock result for visual display if PGVector is down
        return [
            {"score": 0.95, "sqlite_id": 99, "category": "Bags", "color": "Red", "found_description": "Mock Red Backpack, broken zipper.", "image_path": MOCK_IMAGE_URL},
            {"score": 0.88, "sqlite_id": 98, "category": "Electronics", "color": "Black", "found_description": "Mock Black Phone, cracked screen.", "image_path": MOCK_IMAGE_URL}
        ]

    # 1. Extract Structured Filters (from standardized LOST item JSON)
    category_filter = lost_item_json.get("item_category")
    
    langchain_filter = {}
    
    # Only apply filter if the category is valid (not "null" or empty)
    if category_filter and category_filter != "null":
        # LangChain's filter structure for PGVector often uses simple key-value matching
        # Note: PGVector's filter structure is typically simplified in LangChain
        # We assume exact match for category for initial filtering efficiency
        langchain_filter["item_category"] = category_filter

    # The lost item's full description is the query text for vector comparison
    query_text = lost_item_json.get("description", "")
    if not query_text:
        return []
    
    st.info(f"Hybrid Search Query: '{query_text[:30]}...' | Filter: {langchain_filter}")

    # 2. Execute Hybrid Search (PGVector handles the filtering internally before similarity)
    results_with_score = vector_store.similarity_search_with_score(
        query_text,
        k=k,
        filter=langchain_filter
    )

    # 3. Format results
    matched_items = []
    for doc, score in results_with_score:
        # Note: doc.metadata contains image_path and other tags
        matched_items.append({
            "score": round(score, 4),
            "sqlite_id": doc.metadata.get("sqlite_id", "N/A"),
            "category": doc.metadata.get("item_category", "N/A"),
            "color": doc.metadata.get("color", "N/A"),
            "found_description": doc.page_content,
            # Use mock image URL if path is missing or invalid
            "image_path": doc.metadata.get("image_path", MOCK_IMAGE_URL)
        })
        
    return matched_items


def validate_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", phone))


def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


# =======================
# STREAMLIT SETUP
# =======================

st.set_page_config(
    page_title="Lost and Found Intake",
    page_icon="🧳",
    layout="wide",
)

init_db()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Upload Found Item (Operator)", "Report Lost Item (User)"],
)


# ===============================================================
# OPERATOR SIDE
# ===============================================================

if page == "Upload Found Item (Operator)":
    st.title("Operator View: Upload Found Item")

    tag_data = load_tag_data()
    if not tag_data:
        st.stop()

    if gemini_client is None:
        st.info("Gemini is not available. Automated description is disabled.")
        st.stop()

    # Start operator chat if needed
    if "operator_chat" not in st.session_state:
        st.session_state.operator_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM_PROMPT,
            ),
        )
        st.session_state.operator_msgs = []

    # Show conversation
    for msg in st.session_state.operator_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Initial intake
    if not st.session_state.operator_msgs:
        st.markdown("Start by uploading an image or giving a short description of the found item.")

        col1, col2 = st.columns(2)
        uploaded_image = None
        initial_text = ""
        
        with col1:
            uploaded_image = st.file_uploader(
                "Image of the found item (optional)",
                type=["jpg", "jpeg", "png"],
                key="operator_image",
            )
            # --- MOCK IMAGE PATH ---
            image_path_to_save = f"mock_storage/found_{datetime.now().strftime('%Y%m%d%H%M%S')}.png" if uploaded_image else MOCK_IMAGE_URL
            if uploaded_image:
                 st.image(Image.open(uploaded_image).convert("RGB"), width=200)

        with col2:
            initial_text = st.text_input(
                "Short description",
                placeholder="For example black backpack with a NASA patch",
                key="operator_text",
            )

        if st.button("Start Intake"):
            if not uploaded_image and not initial_text:
                st.error("Please upload an image or enter a short description.")
            else:
                message_content = ""

                if uploaded_image:
                    message_content += "I have a photo of the found item. Here is my description based on what I see: "

                if initial_text:
                    message_content += initial_text

                st.session_state.operator_msgs.append(
                    {"role": "user", "content": message_content}
                )
                with st.spinner("Analyzing item"):
                    # For VLM tasks, you should pass the image data to Gemini, 
                    # but for chat-only LLM (gemini-2.0-flash) we send the text description.
                    response = safe_send(
                        st.session_state.operator_chat,
                        message_content,
                        context="operator intake",
                    )
                st.session_state.operator_msgs.append(
                    {"role": "model", "content": response.text}
                )
                
                # Store the image path temporarily for saving later
                st.session_state.operator_image_path = image_path_to_save 
                st.rerun()

    # Continue chat
    operator_input = st.chat_input("Add more details or say 'done' when ready")
    if operator_input:
        st.session_state.operator_msgs.append(
            {"role": "user", "content": operator_input}
        )
        with st.spinner("Processing"):
            response = safe_send(
                st.session_state.operator_chat,
                operator_input,
                context="operator follow-up",
            )
        st.session_state.operator_msgs.append(
            {"role": "model", "content": response.text}
        )
        st.rerun()

    # Check for final structured record
    if st.session_state.operator_msgs and is_structured_record(
        st.session_state.operator_msgs[-1]["content"]
    ):
        structured_text = st.session_state.operator_msgs[-1]["content"]
        st.markdown("### Final structured description")
        st.code(structured_text)

        final_json = standardize_description(structured_text, tag_data)
        if final_json:
            st.success("Standardized JSON")
            st.json(final_json)

            contact = st.text_input("Operator contact or badge")
            if st.button("Save Found Item"):
                if not contact:
                    st.error("Please enter operator contact.")
                else:
                    location_value = (
                        final_json["subway_location"][0]
                        if final_json.get("subway_location")
                        else ""
                    )
                    success = add_found_item(
                        final_json.get("description", ""),
                        location_value,
                        contact,
                        st.session_state.get("operator_image_path", MOCK_IMAGE_URL), # Use path from session state
                        json.dumps(final_json),
                    )
                    if success:
                         st.session_state.operator_msgs = [] # Clear chat on success
                         st.rerun()


# ===============================================================
# USER SIDE: INTEGRATED MATCHING
# ===============================================================

if page == "Report Lost Item (User)":
    st.title("Report Lost Item (User) - Matching Enabled")

    tag_data = load_tag_data()
    if not tag_data:
        st.stop()

    if gemini_client is None:
        st.info("Gemini is not available. Automated structuring is disabled.")
        st.stop()

    st.markdown("You can give quick info using dropdowns, then refine with chat.")

    # --- Quick Info Dropdowns ---
    with st.expander("Optional quick info"):
        col1, col2, col3 = st.columns(3)
        with col1:
            location_choice = st.selectbox(
                "Subway station (optional)", [""] + tag_data["locations"]
            )
        with col2:
            category_choice = st.selectbox(
                "Item category (optional)", [""] + tag_data["categories"]
            )
        with col3:
            type_choice = st.selectbox(
                "Item type (optional)", [""] + tag_data["item_types"]
            )
    # ---------------------------

    st.subheader("Describe or show your lost item")

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
            placeholder="For example blue iPhone with cracked screen",
            key="user_text",
        )

    # Start user chat if needed
    if "user_chat" not in st.session_state:
        st.session_state.user_chat = gemini_client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=USER_SIDE_GENERATOR_PROMPT,
            ),
        )
        st.session_state.user_msgs = []

    # Show chat history
    for msg in st.session_state.user_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Start report
    if not st.session_state.user_msgs and st.button("Start Report"):
        if not uploaded_image and not initial_text:
            st.error("Please upload an image or enter a short description.")
        else:
            message_text = ""
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, width=250)
                message_text += "I have uploaded an image of my lost item. "

            if initial_text:
                message_text += initial_text

            st.session_state.user_msgs.append(
                {"role": "user", "content": message_text}
            )
            with st.spinner("Analyzing"):
                # NOTE: If using a VLM here (like gemini-2.5-flash-preview-09-2025), 
                # you should pass the image object/bytes in the contents list.
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
    user_input = st.chat_input("Add more details or say 'done' when ready")
    if user_input:
        st.session_state.user_msgs.append(
            {"role": "user", "content": user_input}
        )
        with st.spinner("Thinking"):
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
Item Type: {type_choice or extract_field(structured_text, 'Item Type')}<br>
Description: {extract_field(structured_text, 'Description')}
        """

        st.markdown("### 1. Standardizing Record")
        final_json = standardize_description(merged_text, tag_data)
        if final_json:
            st.success("Standardized record created.")
            st.json(final_json)

            # 2. MATCHING WORKSTREAM (Hybrid Search)
            st.markdown("### 2. Searching for Found Items (Hybrid Match)")
            
            # Check if vector search is even possible
            if vector_store is None:
                st.error("Cannot perform matching: Vector store not configured.")
                # Show mock results even if real PGVector isn't running
                matched_items = hybrid_match_found_items(final_json, k=3)
            else:
                with st.spinner("Executing Hybrid Vector Search..."):
                    # Calling the Hybrid Match function
                    matched_items = hybrid_match_found_items(final_json, k=3)
                
            if matched_items:
                st.success(f"Found {len(matched_items)} potential matches!")
                
                # Display the matched items
                for i, match in enumerate(matched_items):
                    st.markdown(f"**Match #{i+1} (Confidence: {match['score']:.4f})**")
                    
                    col_score, col_details, col_image = st.columns([1, 3, 1])
                    
                    with col_details:
                        st.write(f"DB ID: {match['sqlite_id']}")
                        st.caption(f"Category: {match['category']}, Color: {match['color']}")
                        st.write(f"Found Description: {match['found_description']}")
                        
                    with col_image:
                        # Display a mock image based on the path stored in PGVector metadata
                        # Use the placeholder image URL for the mock path
                        img_path = match['image_path'] if match['image_path'] else MOCK_IMAGE_URL
                        st.image(img_path, caption=f"ID: {match['sqlite_id']}")
                        
                    st.markdown("---")
                    
            else:
                st.info("No high-confidence matches found.")


            # 3. CONTACT AND SUBMISSION
            st.markdown("### 3. Contact information")
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

    # Optional vector search debug panel
    with st.expander("Vector search (debug)"):
        if vector_store is None:
            st.info("Vector search is not configured or not available.")
        else:
            query = st.text_input("Search similar descriptions")
            if query:
                with st.spinner("Searching similar items"):
                    try:
                        # Direct vector search without metadata filter
                        results = vector_store.similarity_search_with_score(
                            query, k=3
                        )
                        for doc, score in results:
                            st.write(f"Score: {score:.4f}")
                            st.write(doc.page_content)
                            st.markdown(f"Metadata: {doc.metadata}")
                            st.markdown("---")
                    except Exception as e:
                        st.error(f"Error during vector search: {e}")
