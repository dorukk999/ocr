import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import json
import base64
import concurrent.futures
from PIL import Image
import io

# --- SETTINGS ---
APP_PASSWORD = "SizinBelirlediginizSifre" # You can update this password

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

TARGET_MODEL = "models/gemini-2.5-flash"

def optimize_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
    max_size = 2400 
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=98)
    return buffer.getvalue()

def process_file_backend(file_bytes, unique_filename, incoming_key, target_schema):
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
        
        schema_prompt = f"You are an advanced corporate OCR engine. Output exactly these keys: {json.dumps(target_schema)}."
        response = client.models.generate_content(
            model=TARGET_MODEL,
            contents=[schema_prompt, {"inline_data": {"data": base64_data, "mime_type": mime_type}}],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        extracted_json = json.loads(response.text.strip())
        safe_json = {"Source_File_Name": unique_filename}
        for col in target_schema:
            found_key = next((k for k in extracted_json if k.lower().strip() == col.lower().strip()), None)
            safe_json[col] = extracted_json[found_key] if found_key else "-"
        return safe_json
    except Exception as e:
        return {"Source_File_Name": unique_filename, "Document Type": "FAILED", "Remarks for unclear or doubtful fields": str(e)}

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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(process_file_backend, fb, fn, st.session_state["api_key"], active_schema): fn for fb, fn in files_data}
            for future in concurrent.futures.as_completed(futures):
                raw_results.append(future.result())
                progress_bar.progress(len(raw_results) / total)
                table_placeholder.dataframe(pd.DataFrame(raw_results), use_container_width=True)

        df = pd.DataFrame(raw_results)
        if enable_consolidation and "Full Name" in df.columns:
            df = df.groupby('Full Name').agg(lambda x: ' / '.join(set(x.astype(str).str.strip()))).reset_index()
        
        st.success("🎉 Processing completed successfully!")
        st.download_button("📥 Download CSV Report", df.to_csv(index=False).encode('utf-8'), "Consolidated_Report.csv", "text/csv")

if __name__ == "__main__":
    main()
