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
import re
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
    "Full Name": "the document holder's own given name(s) and surname, exactly as printed near a 'Name' / 'الاسم' label. Never the nationality, country, employer, or job title. Always output this in Latin/English script — if the document only shows the name in Arabic, phonetically transliterate it to Latin letters. Never put Arabic script in this field (Arabic script belongs only in 'Arabic Name'). Always order it as [Given Name(s)] [Surname] (e.g. 'Kali Bahadur Thapa'), never [Surname] [Given Name(s)] — even if a passport's own layout lists the surname field first, reorder it to given-name-first so the same person's name is formatted identically across every document type.",
    "Arabic Name": "the same person's name in Arabic script, next to the Arabic 'الاسم' label. Never the Arabic word for a country/nationality.",
    "Nationality": "the person's country/nationality, next to a 'Nationality' / 'الجنسية' label (e.g. Nepal, India). Never the person's name.",
    "UID / Unified Number": "ONLY fill this in if the document has an explicit label reading 'Unified No.', 'UID', 'Unified Number', or the Arabic 'الرقم الموحد' directly next to the value. This field is an exception to the contextual-inference rule below: never derive, compute, or extract it from the Emirates ID number, card number, chip number, or any part of the MRZ string, even if the digit count matches — those are different fields, not the Unified Number. If there is no such explicit label, always output 'N/A' here, no matter how plausible a nearby number looks.",
    "Issuing Authority": "the name of the organization/government body that issued the document (e.g. 'Federal Authority for Identity & Citizenship, Customs & Port Security', 'MOFA, Department of Passport', 'UAE Ministry of Defense') — usually found in the document's header/logo area. This is NEVER a city or place name like 'Al Ain' — a city where the document was issued is the 'Issuing Place' and belongs only in 'Any other visible document-specific fields', not here.",
    "Personal Number (14 Digits)": "ONLY fill this in if the document has an explicit label reading 'Personal Number' or the Arabic 'الرقم الشخصي' directly next to the value. Never fill it in just because some other number on the document (a card number, security pass number, chip number, etc.) happens to be 14 digits long — matching digit count is not evidence it belongs in this field. If there is no such explicit label, output 'N/A'.",
    "Work Permit Number (9 Digits)": "ONLY fill this in if the document has an explicit label reading 'Work Permit Number' or the Arabic equivalent directly next to the value. Never fill it in just because some other number happens to be 9 digits long. If there is no such explicit label, output 'N/A'.",
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

ARABIC_SCRIPT_RE = re.compile(r'[؀-ۿ]')

def compute_review_flags(safe_json):
    full_name = str(safe_json.get("Full Name", "")).strip()
    flags = []
    if full_name and full_name.lower() not in ("-", "n/a"):
        if ARABIC_SCRIPT_RE.search(full_name):
            flags.append("Full Name contains Arabic script — verify")
        elif difflib.get_close_matches(full_name.lower(), NATIONALITY_WORDS, n=1, cutoff=0.8):
            flags.append("Full Name looks like a nationality — verify")
        elif " " not in full_name:
            flags.append("Full Name is a single word — verify")

    uid = str(safe_json.get("UID / Unified Number", "")).strip()
    emirates_id = str(safe_json.get("Emirates ID Number (784-xxxx-xxxxxxx-x)", "")).strip()
    if uid and uid.lower() not in ("-", "n/a"):
        uid_digits = re.sub(r'\D', '', uid)
        id_digits = re.sub(r'\D', '', emirates_id)
        if uid_digits and id_digits:
            match = difflib.SequenceMatcher(None, uid_digits, id_digits).find_longest_match(0, len(uid_digits), 0, len(id_digits))
            fully_contained = match.size == len(uid_digits) and len(uid_digits) >= 5
            if match.size >= 8 or fully_contained:
                flags.append("UID overlaps with Emirates ID/MRZ digits, likely derived not labeled — verify")

    return "; ".join(flags)

# Buckets a document falls into, used to prefix its columns in the per-person wide view.
DOC_TYPE_BUCKETS = [
    ("Passport", ["passport"]),
    ("Emirates ID", ["resident identity", "emirates id", " eid", "identity card"]),
    ("Security Pass", ["security pass", "defense", "military", "access card", "permit"]),
]

def classify_doc_type(document_type_text, filename):
    text = f"{document_type_text} {filename}".lower()
    for bucket_name, keywords in DOC_TYPE_BUCKETS:
        if any(kw in text for kw in keywords):
            return bucket_name
    return "Other Document"

def build_person_view(df):
    """One row per person (grouped by Full Name), with each document's fields
    prefixed by document type (e.g. 'Passport - Passport Number') so nothing
    from different documents collides in the same column."""
    df = df.copy()
    df["__bucket"] = [classify_doc_type(dt, fn) for dt, fn in zip(df.get("Document Type", ""), df.get("Source_File_Name", ""))]

    shared_cols = ["Full Name", "Arabic Name", "Nationality"]
    per_doc_cols = [c for c in df.columns if c not in shared_cols + ["__bucket", "Source_File_Name"]]

    people = {}
    order = []
    for _, row in df.iterrows():
        name = str(row.get("Full Name", "")).strip()
        # Match names case-insensitively (e.g. "Manish Kumar Sah" vs "MANISH KUMAR SAH" are the same person)
        key = name.lower() if name and name.lower() not in ("-", "n/a") else f"__unmatched__{row.get('Source_File_Name', '')}"
        if key not in people:
            people[key] = {"Full Name": name}
            for c in shared_cols[1:]:
                people[key][c] = row.get(c, "")
            order.append(key)
        for c in shared_cols[1:]:
            if not str(people[key].get(c, "")).strip() or str(people[key][c]).strip().lower() in ("-", "n/a"):
                people[key][c] = row.get(c, "")
        bucket = row["__bucket"]
        for c in per_doc_cols:
            col_name = f"{bucket} - {c}"
            value = row.get(c, "")
            existing = people[key].get(col_name)
            if existing is None or str(existing).strip().lower() in ("", "-", "n/a"):
                people[key][col_name] = value

    return pd.DataFrame([people[k] for k in order])

def build_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Report")
        worksheet = writer.sheets["Report"]
        wrap = Alignment(wrap_text=True, vertical="top")
        for col_idx, col_name in enumerate(df.columns, start=1):
            base_name = col_name.split(" - ", 1)[-1]
            width = COLUMN_WIDTHS.get(col_name, COLUMN_WIDTHS.get(base_name, 18))
            worksheet.column_dimensions[get_column_letter(col_idx)].width = width
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
            "Never substitute a DIFFERENT field's value into a field it doesn't belong to just because it "
            "seems like the closest fit — for example, a passport's Citizenship No. is not a "
            "'UID / Unified Number', and a Place of Birth is not a 'Site Location'. If no genuine counterpart "
            "for a field exists on the document, output 'N/A' for it and put that unrelated leftover value in "
            "'Any other visible document-specific fields' instead.\n"
            "This does NOT mean you should leave a field as 'N/A' when the correct value for THAT SAME field "
            "is clearly identifiable from context or position even without a printed label — e.g. if a "
            "country/nationality word appears where nationality is normally shown, still fill in 'Nationality' "
            "with it, and just note in 'Remarks' that it wasn't explicitly labeled."
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

CHUNK_SIZE = 50
MAX_WORKERS = 5

def process_batch(files_data, api_key, schema, model, progress_bar, table_placeholder, total):
    processed_count = 0
    last_render = 0.0
    for i in range(0, len(files_data), CHUNK_SIZE):
        chunk = files_data[i:i + CHUNK_SIZE]
        chunk_bytes_by_name = {fn: fb for fb, fn in chunk}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_file_backend, fb, fn, api_key, schema, model): fn for fb, fn in chunk}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                st.session_state["batch_results"].append(result)
                processed_count += 1
                filename = result["Source_File_Name"]
                if result.get("Document Type") == "FAILED":
                    st.session_state["failed_bytes"][filename] = chunk_bytes_by_name[filename]
                else:
                    st.session_state["failed_bytes"].pop(filename, None)
                progress_bar.progress(processed_count / total)
                now = time.time()
                if now - last_render > 0.5 or processed_count == total:
                    table_placeholder.dataframe(pd.DataFrame(st.session_state["batch_results"]), use_container_width=True)
                    last_render = now

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
        enable_consolidation = st.checkbox(
            "Enable Person-Based Consolidation",
            help="One row per person instead of one row per document. Each document's fields get their own prefixed columns (e.g. 'Passport - Passport Number', 'Emirates ID - Date of Expiry'), so nothing from different documents collides in the same column."
        )
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.rerun()

    active_schema = OFFICIAL_SCHEMA_ORDER.copy()
    if custom_field: active_schema.insert(19, custom_field)

    if "batch_results" not in st.session_state: st.session_state["batch_results"] = []
    if "failed_bytes" not in st.session_state: st.session_state["failed_bytes"] = {}

    uploaded_files = st.file_uploader("Upload Documents", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

    if uploaded_files and st.session_state["api_key"] and st.button("🚀 Run Extraction Pipeline"):
        st.session_state["batch_results"] = []
        st.session_state["failed_bytes"] = {}
        total = len(uploaded_files)
        progress_bar = st.progress(0)
        table_placeholder = st.empty()
        files_data = [(f.read(), f.name) for f in uploaded_files]
        process_batch(files_data, st.session_state["api_key"], active_schema, selected_model, progress_bar, table_placeholder, total)
        st.success(f"🎉 Processing completed — {len(st.session_state['batch_results'])} file(s) processed.")

    if st.session_state["failed_bytes"]:
        n_failed = len(st.session_state["failed_bytes"])
        st.warning(f"⚠️ {n_failed} file(s) failed after retries.")
        if st.button(f"🔁 Retry {n_failed} Failed File(s) Only"):
            failed_names = set(st.session_state["failed_bytes"].keys())
            st.session_state["batch_results"] = [r for r in st.session_state["batch_results"] if r["Source_File_Name"] not in failed_names]
            files_data = [(fb, fn) for fn, fb in st.session_state["failed_bytes"].items()]
            total = len(files_data)
            progress_bar = st.progress(0)
            table_placeholder = st.empty()
            process_batch(files_data, st.session_state["api_key"], active_schema, selected_model, progress_bar, table_placeholder, total)
            st.rerun()

    if st.session_state["batch_results"]:
        df = pd.DataFrame(st.session_state["batch_results"])
        if enable_consolidation and "Full Name" in df.columns:
            df = build_person_view(df)

        st.download_button(
            "📊 Download Excel Report (.xlsx)",
            build_excel_bytes(df),
            "Consolidated_Report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button("📥 Download CSV Report", df.to_csv(index=False).encode('utf-8-sig'), "Consolidated_Report.csv", "text/csv")

if __name__ == "__main__":
    main()
