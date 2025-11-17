# =======================
# LOST AND FOUND INTAKE SYSTEM
# =======================

import streamlit as st
import sqlite3
import json
import re
from datetime import datetime, timezone
from PIL import Image
import pandas as pd
from google import genai
from google.genai import types


# =======================
# GEMINI CLIENT
# =======================

@st.cache_resource
def get_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error("GEMINI_API_KEY is not set in Streamlit secrets.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None


gemini_client = get_gemini_client()

# =======================
# PROMPTS
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
    """
    Load Tags.xlsx and prepare tag lists.
    Expected columns: Subway Location, Color, Item Category, Item Type.
    """
    try:
        df = pd.read_excel("Tags.xlsx")
        return {
            "df": df,
            "locations": sorted(set(df["Subway Location"].dropna().astype(str))),
            "colors": sorted(set(df["Color"].dropna().astype(str))),
            "categories": sorted(set(df["Item Category"].dropna().astype(str))),
            "item_types": sorted(set(df["Item Type"].dropna().astype(str))),
        }
    except Exception as e:
        st.error(f"Error loading tag data: {e}")
        return None


def extract_field(text: str, field: str) -> str:
    """Extract a simple `Field: value` line from a structured block."""
    match = re.search(rf"{field}:\s*(.*)", text)
    return match.group(1).strip() if match else "null"


def is_structured_record(message: str) -> bool:
    """Detect if the message looks like the final structured record."""
    return message.strip().startswith("Subway Location:")


def standardize_description(text: str, tags: dict) -> dict:
    """
    Send structured text and tag data summary to Gemini for JSON standardization.
    """
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

    try:
    response = gemini_client.models.generate_content(
        model="gemini-1.5-flash",
        contents=full_prompt,
    )
except Exception as e:
    st.error(f"Error calling Gemini for standardization: {e}")
    return {}

    try:
        cleaned = response.text.strip()
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}") + 1
        json_text = cleaned[json_start:json_end]
        data = json.loads(json_text)

        # Ensure required keys exist and fill time if missing
        if "time" not in data or not data["time"]:
            data["time"] = datetime.now(timezone.utc).isoformat()

        # Normalize list type fields
        for key in ["subway_location", "color", "item_type"]:
            if key in data and isinstance(data[key], str):
                data[key] = [data[key]]
            elif key not in data:
                data[key] = []

        if "item_category" not in data:
            data["item_category"] = "null"

        if "description" not in data:
            data["description"] = extract_field(text, "Description")

        return data
    except Exception:
        st.error("Model output could not be parsed as JSON. Raw output is displayed below.")
        st.text(response.text)
        return {}


# =======================
# DATABASE HELPERS
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
    with sqlite3.connect("lost_and_found.db") as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO found_items (caption, location, contact, image_path, json_data)
            VALUES (?, ?, ?, ?, ?)
        """,
            (caption, location, contact, image_path, json_data_string),
        )
        conn.commit()


def add_lost_item(description, contact, email, json_data_string):
    with sqlite3.connect("lost_and_found.db") as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO lost_items (description, contact, email, json_data)
            VALUES (?, ?, ?, ?)
        """,
            (description, contact, email, json_data_string),
        )
        conn.commit()


def validate_phone(phone: str) -> bool:
    return bool(re.fullmatch(r"\d{10}", phone))


def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


# =======================
# STREAMLIT UI
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
        st.info("Gemini is not available. You can still use manual fields but automated description will not run.")
        st.stop()

    if "operator_chat" not in st.session_state:
    st.session_state.operator_chat = gemini_client.chats.create(
        model="gemini-1.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=GENERATOR_SYSTEM_PROMPT,
        ),
    )
    st.session_state.operator_msgs = []

    # Show running conversation
    for msg in st.session_state.operator_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Initial intake
    if not st.session_state.operator_msgs:
        st.markdown("Start by uploading an image or giving a short description of the found item.")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_image = st.file_uploader(
                "Image of the found item (optional)",
                type=["jpg", "jpeg", "png"],
                key="operator_image",
            )
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

                st.session_state.operator_msgs.append(
                    {"role": "user", "content": message_content}
                )
                with st.spinner("Analyzing item"):
                    response = st.session_state.operator_chat.send_message(prompt_parts)
                st.session_state.operator_msgs.append(
                    {"role": "model", "content": response.text}
                )
                st.rerun()

    # Continue chat
    operator_input = st.chat_input("Add more details or say done when ready")
    if operator_input:
        st.session_state.operator_msgs.append(
            {"role": "user", "content": operator_input}
        )
        with st.spinner("Processing"):
            response = st.session_state.operator_chat.send_message(operator_input)
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
                location_value = (
                    final_json["subway_location"][0]
                    if final_json.get("subway_location")
                    else ""
                )
                add_found_item(
                    final_json.get("description", ""),
                    location_value,
                    contact,
                    "",
                    json.dumps(final_json),
                )
                st.success("Found item saved to database.")


# ===============================================================
# USER SIDE
# ===============================================================

if page == "Report Lost Item (User)":
    st.title("User View: Report a Lost Item")

    tag_data = load_tag_data()
    if not tag_data:
        st.stop()

    if gemini_client is None:
        st.info("Gemini is not available. You can still submit details manually, but auto structuring will not run.")
        st.stop()

    st.markdown("You can give quick info using dropdowns, then refine with chat.")

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

    if "user_chat" not in st.session_state:
    st.session_state.user_chat = gemini_client.chats.create(
        model="gemini-1.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=USER_SIDE_GENERATOR_PROMPT,
        ),
    )
    st.session_state.user_msgs = []


    # Show chat history
    for msg in st.session_state.user_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.user_msgs and st.button("Start Report"):
        if not uploaded_image and not initial_text:
            st.error("Please upload an image or enter a short description.")
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

            st.session_state.user_msgs.append(
                {"role": "user", "content": message_text}
            )
            with st.spinner("Analyzing"):
                response = st.session_state.user_chat.send_message(parts)
            st.session_state.user_msgs.append(
                {"role": "model", "content": response.text}
            )
            st.rerun()

    user_input = st.chat_input("Add more details or say done when ready")
    if user_input:
        st.session_state.user_msgs.append(
            {"role": "user", "content": user_input}
        )
        with st.spinner("Thinking"):
            response = st.session_state.user_chat.send_message(user_input)
        st.session_state.user_msgs.append(
            {"role": "model", "content": response.text}
        )
        st.rerun()

    # When final structured record appears
    if st.session_state.user_msgs and is_structured_record(
        st.session_state.user_msgs[-1]["content"]
    ):
        structured_text = st.session_state.user_msgs[-1]["content"]

        merged_text = f"""
Subway Location: {location_choice or extract_field(structured_text, 'Subway Location')}
Color: {extract_field(structured_text, 'Color')}
Item Category: {category_choice or extract_field(structured_text, 'Item Category')}
Item Type: {type_choice or extract_field(structured_text, 'Item Type')}
Description: {extract_field(structured_text, 'Description')}
        """

        st.markdown("### Final merged record before standardization")
        st.code(merged_text)

        final_json = standardize_description(merged_text, tag_data)
        if final_json:
            st.success("Standardized record")
            st.json(final_json)

            st.markdown("### Contact information")
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
                    st.success("Lost item report submitted.")

