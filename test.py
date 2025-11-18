# =======================
# LOST & FOUND INTAKE SYSTEM
# =======================

import streamlit as st
import sqlite3, json, re
from datetime import datetime, timezone
from PIL import Image
import pandas as pd
from google import genai
from typing import List, Tuple, Dict
# Required imports for matching model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- Initialize Gemini client ---
# Ensure you have your GEMINI_API_KEY set in Streamlit secrets
try:
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Error initializing Gemini Client. Check st.secrets['GEMINI_API_KEY'].")
    st.stop()


# =======================
# PROMPTS
# =======================

GENERATOR_SYSTEM_PROMPT = """
Role:
You are a Lost & Found intake operator for a public-transit system. Your job is to gather accurate, factual information
about a lost item, refine the description interactively with the user, and automatically pass the finalized record to a second GPT
for tag standardization.

Behavior Rules:
1. Input Handling
The user may provide either an image or a short text description.
If an image is provided, immediately describe visible traits (color, material, type, size, markings, notable features).
If text is provided, restate and cleanly summarize it in factual language. Do not wait for confirmation before generating the first description.
2. Clarification
After your initial description, ask targeted, concise follow-up questions to collect identifying details such as brand, condition,
writing, contents, location (station), and time lost.
If the user provides a station name (e.g., “Times Sq”, “Queensboro Plaza”), automatically identify the corresponding MTA subway line(s)
and record them in the Subway Location field.
Example: “Times Sq” → Lines 1, 2, 3, 7, N, Q, R, W, S.
If multiple lines serve the station, include all.
However, if the station name has 4 or more subway lines, record only the station name in the Subway Location field.
If the station name is unclear or not found, set Subway Location to null.
As the user answers, update and refine your internal description dynamically.
Stop asking questions once enough information has been gathered for a clear and specific description.
Do not include the questions or intermediate notes in the final output.
3. Finalization
When the user says they are finished, or you have enough detail, generate only the final structured record in this format:
Subway Location: <observed or user-provided line, or null>
Color: <dominant or user-provided color(s), or null>
Item Category: <free-text category, e.g., Bags & Accessories, Electronics, Clothing, or null>
Item Type: <free-text item type, e.g., Backpack, Phone, Jacket, or null>
Description: <concise free-text summary combining all verified details>
"""

USER_SIDE_GENERATOR_PROMPT = """
You are a helpful assistant helping subway riders report lost items.

Input:
The user may provide either an image or a short text description of their item.
If an image is provided, begin by describing what you see — include visible traits such as color, material, size, shape, and any markings or distinctive details.
If text is provided, restate their message cleanly in factual language.

Clarification:
Then, ask 2–4 concise follow-up questions to collect identifying details such as:
- color (if not already clear from the image),
- brand or logo,
- contents (if a bag or container),
- any markings or writing,
- where it was lost (station name),
- and approximate time.

When you have enough details, output ONLY the structured record in this format:

Subway Location: <station name or null>
Color: <color(s) or null>
Item Category: <category or null>
Item Type: <type or null>
Description: <concise factual summary combining all verified details>

Guidelines:
- Keep your tone concise and factual.
- Do not include your reasoning or notes.
- Do not output questions or conversation history in the final structured record.
"""

STANDARDIZER_PROMPT = """
You are the Lost & Found Data Standardizer for a public-transit system. You receive structured text from another model describing a lost item. Your job is to map free-text fields to standardized tag values and produce a clean JSON record ready for database storage.

Data Source:
All valid standardized values are stored in the Tags Excel reference file uploaded.
This file is the only source of truth for all mappings.
The Tags Excel contains separate tabs or columns for the following standardized lists:

Subway Location → All valid subway lines and station names
Item Category → All valid item category names
Item Type → All valid item type names
Color → All valid color names

When standardizing input text:

Always match each field only against its corresponding tag list:
subway_location → compare only with values in Subway Location
color → compare only with values in Color
item_category → compare only with values in Item Category
item_type → compare only with values in Item Type
Never mix across tag types.
Use exact or closest textual matches from the relevant tag column only.
If no valid match is found, return "null".
Output the standardized value exactly as it appears in the Excel file — no prefixes, suffixes, or formatting changes.

Behavior Rules:
1. Input Format:
You will receive input in this structure:
Subway Location: <value or null>
Color: <value or null>
Item Category: <value or null>
Item Type: <value or null>
Description: <free-text description>

2. Standardization:
Use the provided Tags Excel reference to ensure consistent value mapping.
Subway Location: Match to valid MTA lines or stations. If none or unclear, output "null".
Color: Match to standardized color names. If multiple colors appear, include all as an array.
Item Category: Map to a consistent category.
Item Type: Map to consistent type(s). If multiple types appear, include all as an array.
Description: Leave as free text but clean it up.
Time: Record the current system time (ISO 8601 UTC).

3. Output Format:
Produce only a JSON object (no explanations):
{
  "subway_location": ["<line1>", "<line2>"],
  "color": ["<color1>", "<color2>"],
  "item_category": "<standardized category or null>",
  "item_type": ["<standardized type>", "<standardized type>"],
  "description": "<clean final description>",
  "time": "<ISO 8601 UTC timestamp>"
}

4. Behavior Guidelines:
- Do not guess missing details.
- If uncertain, leave field as null.
- Ensure valid JSON output.
- Only output the JSON object, nothing else.
"""

# =======================
# DATA HELPERS
# =======================

@st.cache_data
def load_tag_data():
    """Loads standardized tag data from Tags.xlsx."""
    try:
        # Assuming Tags.xlsx is available in the current directory
        df = pd.read_excel("Tags.xlsx") 
        return {
            "df": df,
            "locations": sorted(set(df["Subway Location"].dropna())),
            "colors": sorted(set(df["Color"].dropna())),
            "categories": sorted(set(df["Item Category"].dropna())),
            "item_types": sorted(set(df["Item Type"].dropna()))
        }
    except Exception as e:
        st.error(f"Error loading tag data: {e}. Ensure 'Tags.xlsx' is present.")
        return None


def extract_field(text, field):
    """Extracts a field value from the structured text output."""
    match = re.search(rf"{field}:\s*(.*)", text)
    return match.group(1).strip() if match else "null"

def standardize_description(text, tags):
    """Send structured text + tag data to Gemini for JSON standardization."""
    tags_summary = (
        f"\n--- TAGS REFERENCE ---\n"
        f"Subway Location tags: {', '.join(tags['locations'][:50])}\n"
        f"Color tags: {', '.join(tags['colors'][:50])}\n"
        f"Item Category tags: {', '.join(tags['categories'][:50])}\n"
        f"Item Type tags: {', '.join(tags['item_types'][:50])}\n"
    )

    full_prompt = f"{STANDARDIZER_PROMPT}\n\nHere is the structured input to standardize:\n{text}\n{tags_summary}"

    model = gemini_client.models.get("gemini-1.5-flash")
    response = model.generate_content(full_prompt)
    try:
        # Extract valid JSON from model output
        cleaned = response.text.strip()
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        json_text = cleaned[json_start:json_end]
        return json.loads(json_text)
    except Exception:
        st.error("Model output could not be parsed as JSON. Displaying raw output:")
        st.text(response.text)
        return {}

# =======================
# MATCHING MODEL
# =======================

@st.cache_data
def get_top_matches(
    item_list: List[Tuple[int, str, str]], 
    lost_item_data: Dict[str, any]
) -> List[str]:
    """
    Computes cosine similarity only between the lost item's *Description* and all pre-filtered found item descriptions, returning the top 3 image paths.
    
    item_list format: [(unique_id, image_path, description), ...]
    """
    
    # 1. Prepare data for vectorization - ONLY using the 'Description' field
    
    # Extract the lost item's main description from the JSON
    query_description = lost_item_data.get('description', '') # Note the lowercase key
    
    # Extract only the description (index 2) and image path (index 1) from the database results
    image_paths = [item[1] for item in item_list]
    item_descriptions = [item[2] for item in item_list]
    
    # Combine the query description with all item descriptions for vectorization
    all_descriptions = [query_description] + item_descriptions
    
    # 2. Compute Cosine Similarity (Vectorization)
    
    # Initialize and fit the TfidfVectorizer to the combined corpus
    vectorizer = TfidfVectorizer().fit(all_descriptions)
    
    # Transform all descriptions into TF-IDF vectors
    tfidf_matrix = vectorizer.transform(all_descriptions)
    
    # Separate the query vector (first row) from the item vectors (remaining rows)
    query_vector = tfidf_matrix[0:1]
    item_vectors = tfidf_matrix[1:]
    
    # Calculate the cosine similarity between the query vector and all item vectors
    cosine_scores = cosine_similarity(query_vector, item_vectors).flatten()
    
    # 3. Rank and Retrieve Top 3
    
    # Create a list of (score, image_path) tuples
    scored_matches = []
    for i, score in enumerate(cosine_scores):
        scored_matches.append((score, image_paths[i]))
        
    # Sort the list by score in descending order
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    
    # Get the top 3 image paths
    top_3_paths = [path for score, path in scored_matches[:3]]
    
    return top_3_paths


# =======================
# DATABASE HELPERS
# =======================

def init_db():
    conn = sqlite3.connect("lost_and_found.db")
    c = conn.cursor()
    # Found Items table includes the standardized JSON data
    c.execute("""
        CREATE TABLE IF NOT EXISTS found_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subway_location TEXT,
            color TEXT,
            item_category TEXT,
            item_type TEXT,
            description TEXT,
            contact TEXT,
            image_path TEXT,
            json_data TEXT
        )
    """)
    # Lost Items table for reports
    c.execute("""
        CREATE TABLE IF NOT EXISTS lost_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT,
            contact TEXT,
            email TEXT,
            json_data TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_found_item(json_data: dict, contact: str, image_path: str = ""):
    """Adds a found item record, extracting key fields from the JSON."""
    conn = sqlite3.connect("lost_and_found.db")
    c = conn.cursor()
    # The JSON output contains lists for location, color, and type. We store them as comma-separated strings for easy SQL use/display.
    c.execute(
        """
        INSERT INTO found_items (
            subway_location, color, item_category, item_type, description, contact, image_path, json_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ", ".join(json_data.get("subway_location", [])), # Stored as string
            ", ".join(json_data.get("color", [])), # Stored as string
            json_data.get("item_category", ""),
            ", ".join(json_data.get("item_type", [])), # Stored as string
            json_data.get("description", ""),
            contact,
            image_path,
            json.dumps(json_data)
        )
    )
    conn.commit()
    conn.close()

def add_lost_item(description, contact, email, json_data_string):
    conn = sqlite3.connect("lost_and_found.db")
    c = conn.cursor()
    c.execute("INSERT INTO lost_items (description, contact, email, json_data) VALUES (?, ?, ?, ?)",
              (description, contact, email, json_data_string))
    conn.commit()
    conn.close()


# =======================
# STREAMLIT UI
# =======================

st.set_page_config(page_title="Lost & Found Intake", page_icon="🧳", layout="wide")
init_db()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Upload Found Item (Operator)", "Report Lost Item (User)"])


# ===============================================================
# OPERATOR SIDE (CHAT INTAKE)
# ===============================================================

if page == "Upload Found Item (Operator)":
    st.title("Operator: Describe Found Item")

    tag_data = load_tag_data()
    if not tag_data: st.stop()

    if "operator_chat" not in st.session_state:
        st.session_state.operator_chat = gemini_client.models.start_chat(
            system_instruction=GENERATOR_SYSTEM_PROMPT
        )
        st.session_state.operator_msgs = []
        st.session_state.operator_done = False

    for msg in st.session_state.operator_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Initial input
    if not st.session_state.operator_msgs:
        uploaded_image = st.file_uploader("Upload an image of the found item (optional)", type=["jpg","jpeg","png"])
        initial_text = st.text_input("Or briefly describe the item")

        if st.button("Start Intake"):
            if not uploaded_image and not initial_text:
                st.error("Please upload or describe the item.")
            else:
                prompt_parts = []
                message_content = ""
                if uploaded_image:
                    img = Image.open(uploaded_image).convert("RGB")
                    st.image(img, width=200)
                    prompt_parts.append(img)
                    message_content += "Image uploaded.\n"
                if initial_text:
                    prompt_parts.append(initial_text)
                    message_content += initial_text

                st.session_state.operator_msgs.append({"role":"user","content":message_content})
                with st.spinner("Analyzing..."):
                    response = st.session_state.operator_chat.send_message(prompt_parts)
                st.session_state.operator_msgs.append({"role":"model","content":response.text})
                st.rerun()

    if operator_prompt := st.chat_input("Add more details..."):
        st.session_state.operator_msgs.append({"role":"user","content":operator_prompt})
        with st.spinner("Processing..."):
            response = st.session_state.operator_chat.send_message(operator_prompt)
        st.session_state.operator_msgs.append({"role":"model","content":response.text})
        st.rerun()

    # Detect final structured text
    final_json = {}
    if st.session_state.operator_msgs and st.session_state.operator_msgs[-1]["content"].startswith("Subway Location:"):
        structured_text = st.session_state.operator_msgs[-1]["content"]
        final_json = standardize_description(structured_text, tag_data)
        st.success("Structured description generated:")
        st.json(final_json)
        
        # Check if an image was uploaded in the session
        image_path = "" # You would typically save the image and get a path here
        
        contact = st.text_input("Operator Contact/Badge ID")
        if st.button("Save Found Item"):
            # Use the corrected add_found_item function
            add_found_item(final_json, contact, image_path)
            st.success("Saved successfully!")


# ===============================================================
# USER SIDE (HYBRID DROPDOWN + CHAT)
# ===============================================================

if page == "Report Lost Item (User)":
    st.title("Report Your Lost Item")

    tag_data = load_tag_data()
    if not tag_data: 
        st.stop()

    # Step 1: Quick dropdowns (excluding color)
    with st.expander("Quick Info (Optional)"):
        # The stored tag data is simplified, using the select box values for merge later
        location = st.selectbox("Subway Station", [""] + tag_data["locations"])
        category = st.selectbox("Item Category", [""] + tag_data["categories"])
        item_type = st.selectbox("Item Type", [""] + tag_data["item_types"])

    # Step 2: Image upload or text start
    st.subheader("Describe or Show Your Lost Item")
    uploaded_image = st.file_uploader("Upload an image of your lost item (optional)", type=["jpg","jpeg","png"])
    initial_text = st.text_input("Or describe it briefly (e.g., 'a red leather backpack with gold zippers')")

    if "user_chat" not in st.session_state:
        st.session_state.user_chat = gemini_client.models.start_chat(
            system_instruction=USER_SIDE_GENERATOR_PROMPT
        )
        st.session_state.user_msgs = []
        st.session_state.user_done = False

    # Display chat history
    for msg in st.session_state.user_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Start the chat ---
    if not st.session_state.user_msgs and st.button("Start Report"):
        if not uploaded_image and not initial_text:
            st.error("Please upload an image or provide a description.")
        else:
            parts = []
            message_text = ""
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, width=250)
                parts.append(image)
                message_text += "Here is an image of my lost item.\n"
            if initial_text:
                parts.append(initial_text)
                message_text += initial_text
            st.session_state.user_msgs.append({"role": "user", "content": message_text})
            with st.spinner("Analyzing..."):
                response = st.session_state.user_chat.send_message(parts)
            st.session_state.user_msgs.append({"role": "model", "content": response.text})
            st.rerun()

    # Continue conversation
    if user_input := st.chat_input("Add more details..."):
        st.session_state.user_msgs.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = st.session_state.user_chat.send_message(user_input)
        st.session_state.user_msgs.append({"role": "model", "content": response.text})
        st.rerun()

    # Step 3: When final structured record appears
    final_json = {}
    if st.session_state.user_msgs and st.session_state.user_msgs[-1]["content"].startswith("Subway Location:"):
        structured_text = st.session_state.user_msgs[-1]["content"]
        # Merge dropdowns with chat output (dropdown takes priority if selected)
        merged_text = f"""
        Subway Location: {location or extract_field(structured_text, 'Subway Location')}
        Color: {extract_field(structured_text, 'Color')}
        Item Category: {category or extract_field(structured_text, 'Item Category')}
        Item Type: {item_type or extract_field(structured_text, 'Item Type')}
        Description: {extract_field(structured_text, 'Description')}
        """

        final_json = standardize_description(merged_text, tag_data)
        st.success("Standardized Record:")
        st.json(final_json)

        contact = st.text_input("Contact Number (10 digits)")
        email = st.text_input("Email Address")
        
        # --- Submit and Match Button ---
        if st.button("Submit Lost Item Report & Find Matches"):
            if not contact or not email:
                st.error("Please provide contact info.")
            else:
                # -----------------------------------------------------------
                # CORE MATCHING LOGIC (SQL FILTER + COSINE SIMILARITY)
                # -----------------------------------------------------------
                
                st.info("Searching for potential matches...")

                # 1. Prepare data for SQL query
                # Use the standardized values which are lists/strings
                colors = final_json["color"]
                category = final_json["item_category"]
                item_types = final_json["item_type"]
                subway_locations = final_json["subway_location"] 
                
                # We need to search the database columns (stored as strings) 
                # against the list of colors/types/locations from the lost item.
                # A robust approach would be to use separate tables for tags, 
                # but for this structure, we search for IN a contained string.
                # Since the found items are stored as comma-separated strings (e.g., "Red, Black")
                # we must use a series of LIKE clauses.

                conn = sqlite3.connect('lost_and_found.db')
                cursor = conn.cursor()

                # Build dynamic LIKE clauses for multiple colors/types/locations
                like_clauses = []
                params = []
                
                # Match against category (stored as a single string)
                if category:
                    like_clauses.append("item_category = ?")
                    params.append(category)
                
                # Match against colors (must check if the found item contains *any* of the lost item's colors)
                if colors:
                    color_likes = [f"color LIKE ?" for _ in colors]
                    like_clauses.append(f"({' OR '.join(color_likes)})")
                    params.extend([f"%{c}%" for c in colors]) # e.g., ["%Red%", "%Black%"]

                # Match against item types
                if item_types:
                    type_likes = [f"item_type LIKE ?" for _ in item_types]
                    like_clauses.append(f"({' OR '.join(type_likes)})")
                    params.extend([f"%{t}%" for t in item_types])

                # Match against subway locations
                if subway_locations:
                    location_likes = [f"subway_location LIKE ?" for _ in subway_locations]
                    like_clauses.append(f"({' OR '.join(location_likes)})")
                    params.extend([f"%{l}%" for l in subway_locations])
                    
                where_clause = " AND ".join(like_clauses)
                
                sql_query = f"""
                SELECT id, image_path, json_data
                FROM found_items
                WHERE {where_clause};
                """
                
                # 4. Execute the query
                try:
                    cursor.execute(sql_query, tuple(params))
                    raw_results = cursor.fetchall()
                except sqlite3.OperationalError as e:
                    st.error(f"SQL Error during filtering: {e}")
                    raw_results = []

                # 5. Transform results for the matching function
                results_for_matching = []
                for id, image_path, json_data_str in raw_results:
                    try:
                        found_item_json = json.loads(json_data_str)
                        # Tuple format: (unique_id, image_path, description)
                        results_for_matching.append((id, image_path, found_item_json.get("description", "")))
                    except json.JSONDecodeError:
                        # Skip corrupted records
                        continue

                conn.close()

                # Record the lost item report
                add_lost_item(final_json["description"], contact, email, json.dumps(final_json))
                st.success("Lost item report submitted successfully!")

                # 6. Run the Cosine Similarity Matching
                if results_for_matching:
                    st.subheader("Potential Matches Found (Ranked by Description Similarity) 👇")
                    
                    # NOTE: We pass the 'final_json' as the lost item data
                    top_paths = get_top_matches(results_for_matching, final_json)
                    
                    # 7. Display the results
                    st.json(top_paths)
                    st.write("---")
                    st.markdown(f"**Top 3 Matches:**")
                    for path in top_paths:
                        st.markdown(f"* `{path}`")
                else:
                    st.info("No initial matches found in the database based on basic tags. Your report has still been saved.")
                # -----------------------------------------------------------











