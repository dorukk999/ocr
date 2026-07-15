import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
import json
import base64
import concurrent.futures
import difflib
import time
from PIL import Image
import io

# --- SETTINGS ---
APP_PASSWORD = "DataOCR_98123606513" # You can update this password

# Page Configuration
st.set_page_config(page_title="AI Document Extraction Tool", layout="wide")

# Initialize Session State
if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
if "api_key" not in st.session_state: st.session_state["api_key"] = ""
if "uploader_key" not in st.session_state: st.session_state["uploader_key"] = 0

# Master Corporate Schema
OFFICIAL_SCHEMA_ORDER = [
    "Document Type", "Full Name", "Arabic Name", "Nationality",
    "Emirates ID Number (784-xxxx-xxxxxxx-x)", "Passport Number",
    "Work Permit Number (9 Digits)", "Personal Number (14 Digits)",
    "Permit / Security Pass / Card Number", "UID / Unified Number",
    "Date of Birth", "Date of Issue", "Date of Expiry", 
    "Employer / Company Name", "Occupation / Trade", "Site Location", 
    "Arabic Site Location Names", "English Translation of Arabic Site Locations", 
    "Issuing Authority", "Any other visible document-specific fields", 
    "Confidence level for each extracted field", "Remarks for unclear or doubtful fields"
]

MAX_RETRIES = 3

# Fields the model has been observed to confuse with one another; give it explicit, disambiguating rules.
FIELD_HINTS = {
    "Full Name": "the document holder's own given name(s) and surname, exactly as printed near a 'Name' / 'الاسم' label. Never the nationality, country, employer, or job title.",
    "Arabic Name": "the same person's name in Arabic script, next to the Arabic 'الاسم' label. Never the Arabic word for a country/nationality.",
    "Nationality": "the person's country/nationality, next to a 'Nationality' / 'الجنسية' label (e.g. Nepal, India). Never the person's name.",
}

# Common nationality/demonym words seen on UAE labor/ID documents — used to flag a Full Name
# that looks like it was actually swapped with the Nationality field.
NATIONALITY_WORDS = {
    "nepal", "nepali", "nepalese", "india", "indian", "pakistan", "pakistani",
    "bangladesh", "bangladeshi", "philippines", "filipino", "filipina",
    "sri lanka", "sri lankan", "egypt", "egyptian", "sudan", "sudanese",
    "syria", "syrian", "jordan", "jordanian", "yemen", "yemeni", "lebanon",
    "lebanese", "iraq", "iraqi", "iran", "iranian", "afghanistan", "afghan",
    "ethiopia", "ethiopian", "kenya", "kenyan", "uganda", "ugandan", "somalia",
    "somali", "morocco", "moroccan", "tunisia", "tunisian", "indonesia",
    "indonesian", "bhutan", "bhutanese", "myanmar", "burmese", "china",
    "chinese", "uae", "emirati", "saudi", "saudi arabian", "oman", "omani",
    "kuwait", "kuwaiti", "qatar", "qatari", "bahrain", "bahraini", "britain",
    "british", "america", "american", "canada", "canadian", "australia",
    "australian", "nigeria", "nigerian", "ghana", "ghanaian", "south africa",
    "south african",
}

COLUMN_WIDTHS = {
    "Source_File_Name": 18, "Document Type": 20, "Full Name": 22,
    "Arabic Name": 22, "Nationality": 14,
    "Emirates ID Number (784-xxxx-xxxxxxx-x)": 24, "Passport Number": 16,
    "Work Permit Number (9 Digits)": 16, "Personal Number (14 Digits)": 16,
    "Permit / Security Pass / Card Number": 20, "UID / Unified Number": 18,
    "Date of Birth": 14, "Date of Issue": 14, "Date of Expiry": 14,
    "Employer / Company Name": 32, "Occupation / Trade": 18,
    "Site Location": 30, "Arabic Site Location Names": 30,
    "English Translation of Arabic Site Locations": 32, "Issuing Authority": 30,
    "Any other visible document-specific fields": 40,
    "Confidence level for each extracted field": 40,
    "Remarks for unclear or doubtful fields": 40, "Review Flags": 30,
    "Processing Time (s)": 12,
}

def compute_review_flags(safe_json):
    full_name = str(safe_json.get("Full Name", "")).strip()
    flags = []
    if full_name and full_name.lower() not in ("-", "n/a"):
        if difflib.get_close_matches(full_name.lower(), NATIONALITY_WORDS, n=1, cutoff=0.8):
            flags.append("Full Name looks like a nationality — verify")
        elif " " not in full_name:
            flags.append("Full Name is a single word — verify")
    return "; ".join(flags)

def build_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Report")
        worksheet = writer.sheets["Report"]
        wrap = Alignment(wrap_text=True, vertical="top")
        for col_idx, col_name in enumerate(df.columns, start=1):
            worksheet.column_dimensions[get_column_letter(col_idx)].width = COLUMN_WIDTHS.get(col_name, 18)
        for row in worksheet.iter_rows(min_row=1, max_row=worksheet.max_row):
            for cell in row:
                cell.alignment = wrap
        for i in range(2, worksheet.max_row + 1):
            worksheet.row_dimensions[i].height = 60
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
        worksheet.freeze_panes = "A2"
    return buffer.getvalue()

def optimize_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
    max_size = 1600
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()

def process_file_backend(file_bytes, unique_filename, incoming_key, target_schema, target_model):
    start_time = time.time()
    try:
        client = genai.Client(api_key=incoming_key)
        is_pdf = unique_filename.lower().endswith('.pdf')
        if is_pdf:
            base64_data = base64.b64encode(file_bytes).decode("utf-8")
            mime_type = "application/pdf"
        else:
            optimized_bytes = optimize_image(file_bytes)
            base64_data = base64.b64encode(optimized_bytes).decode("utf-8")
            mime_type = "image/jpeg"

        field_rules = "\n".join(f"- {k}: {v}" for k, v in FIELD_HINTS.items() if k in target_schema)
        schema_prompt = (
            "You are an advanced corporate OCR engine. "
            f"Output exactly these keys: {json.dumps(target_schema)}. "
            "Every value must be a plain string (or number/date as text) — never a nested object or dict. "
            "Do not attach per-field confidence or remarks inside a value; use the dedicated "
            "'Confidence level for each extracted field' and 'Remarks for unclear or doubtful fields' keys for that instead.\n"
            f"Field-specific rules:\n{field_rules}\n"
            "Never copy a value into the wrong field just because it sits next to a similar-looking label. "
            "If a field has no explicitly labeled counterpart on the document, output 'N/A' for it — "
            "do not substitute a different identifier or value just because it seems like the closest fit "
            "(for example, do not put a passport's Citizenship No. into 'UID / Unified Number', and do not put "
            "a Place of Birth into 'Site Location'). Any such leftover, unmapped values belong only in "
            "'Any other visible document-specific fields', never forced into an unrelated schema key."
        )

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=target_model,
                    contents=[schema_prompt, {"inline_data": {"data": base64_data, "mime_type": mime_type}}],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                extracted_json = json.loads(response.text.strip())
                safe_json = {"Source_File_Name": unique_filename}
                for col in target_schema:
                    found_key = next((k for k in extracted_json if k.lower().strip() == col.lower().strip()), None)
                    value = extracted_json[found_key] if found_key else "-"
                    if isinstance(value, dict):
                        value = value.get("value", value)
                    safe_json[col] = value
                safe_json["Review Flags"] = compute_review_flags(safe_json)
                safe_json["Processing Time (s)"] = round(time.time() - start_time, 1)
                return safe_json
            except genai_errors.APIError as e:
                last_error = e
                is_retryable = e.code in (429, 500, 503)
                if is_retryable and attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last_error
    except Exception as e:
        return {
            "Source_File_Name": unique_filename,
            "Document Type": "FAILED",
            "Remarks for unclear or doubtful fields": str(e),
            "Processing Time (s)": round(time.time() - start_time, 1),
        }

def main():
    st.title("📂 Corporate AI Data Extraction Tool")

    # --- 1. STEP: APP LOGIN ---
    if not st.session_state["authenticated"]:
        st.subheader("🔒 Enter Application Password")
        password_input = st.text_input("Password:", type="password")
        if st.button("Login"):
            if password_input == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password!")
        return

    # --- 2. STEP: API SETTINGS & MAIN OPERATIONS ---
    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state["api_key"] = st.text_input("Enter your Gemini API Key:", type="password", value=st.session_state["api_key"])
        selected_model = st.selectbox(
            "Model",
            ["models/gemini-2.5-flash", "models/gemini-2.5-flash-lite"],
            index=0,
            help="flash-lite is faster and cheaper; test accuracy on your documents before relying on it."
        )
        custom_field = st.text_input("➕ Extract Custom Field")
        enable_consolidation = st.checkbox("Enable Person-Based Consolidation")
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.rerun()

    active_schema = OFFICIAL_SCHEMA_ORDER.copy()
    if custom_field: active_schema.insert(19, custom_field)

    uploaded_files = st.file_uploader("Upload Documents", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.session_state["api_key"] and st.button("🚀 Run Extraction Pipeline"):
        raw_results = []
        total = len(uploaded_files)
        progress_bar = st.progress(0)
        table_placeholder = st.empty()
        files_data = [(f.read(), f.name) for f in uploaded_files]
        
        last_render = 0.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(process_file_backend, fb, fn, st.session_state["api_key"], active_schema, selected_model): fn for fb, fn in files_data}
            for future in concurrent.futures.as_completed(futures):
                raw_results.append(future.result())
                progress_bar.progress(len(raw_results) / total)
                now = time.time()
                if now - last_render > 0.5 or len(raw_results) == total:
                    table_placeholder.dataframe(pd.DataFrame(raw_results), use_container_width=True)
                    last_render = now

        df = pd.DataFrame(raw_results)
        if enable_consolidation and "Full Name" in df.columns:
            df = df.groupby('Full Name').agg(lambda x: ' / '.join(set(x.astype(str).str.strip()))).reset_index()
        
        st.success("🎉 Processing completed successfully!")
        st.download_button(
            "📊 Download Excel Report (.xlsx)",
            build_excel_bytes(df),
            "Consolidated_Report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button("📥 Download CSV Report", df.to_csv(index=False).encode('utf-8-sig'), "Consolidated_Report.csv", "text/csv")

if __name__ == "__main__":
    main()
