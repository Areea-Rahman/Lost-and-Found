# =======================
# LOST & FOUND INTAKE SYSTEM
# =======================
pip install google-genai
import streamlit as st
import sqlite3, json, re
from datetime import datetime, timezone
from PIL import Image
import pandas as pd
from google import genai

# --- Initialize Gemini client ---
gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

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
    try:
        df = pd.read_excel("Tags.xlsx")
        return {
            "df": df,
            "locations": sorted(set(df["Subway Location"].dropna())),
            "colors": sorted(set(df["Color"].dropna())),
            "categories": sorted(set(df["Item Category"].dropna())),
            "item_types": sorted(set(df["Item Type"].dropna()))
        }
    except Exception as e:
        st.error(f"Error loading tag data: {e}")
        return None


def extract_field(text, field):
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
# DATABASE HELPERS
# =======================

def init_db():
    conn = sqlite3.connect("lost_and_found.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS found_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            caption TEXT,
            location TEXT,
            contact TEXT,
            image_path TEXT,
            json_data TEXT
        )
    """)
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

def add_found_item(caption, location, contact, image_path, json_data_string):
    conn = sqlite3.connect("lost_and_found.db")
    c = conn.cursor()
    c.execute("INSERT INTO found_items (caption, location, contact, image_path, json_data) VALUES (?, ?, ?, ?, ?)",
              (caption, location, contact, image_path, json_data_string))
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
    if st.session_state.operator_msgs and st.session_state.operator_msgs[-1]["content"].startswith("Subway Location:"):
        structured_text = st.session_state.operator_msgs[-1]["content"]
        final_json = standardize_description(structured_text, tag_data)
        st.success("Structured description generated:")
        st.json(final_json)
        contact = st.text_input("Operator Contact/Badge ID")
        if st.button("Save Found Item"):
            add_found_item(final_json["description"], final_json["subway_location"][0],
                           contact, "", json.dumps(final_json))
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
    if st.session_state.user_msgs and st.session_state.user_msgs[-1]["content"].startswith("Subway Location:"):
        structured_text = st.session_state.user_msgs[-1]["content"]
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
        if st.button("Submit Lost Item Report"):
            if not contact or not email:
                st.error("Please provide contact info.")
            else:
                add_lost_item(final_json["description"], contact, email, json.dumps(final_json))
                st.success("Lost item report submitted successfully!")


