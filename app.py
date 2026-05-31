import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import json
import base64
import concurrent.futures
from PIL import Image
import io

# Page Configuration
st.set_page_config(page_title="AI Powered Document Extraction Tool", layout="wide")

# Initialize Session State for File Uploader Key to enable global clear
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# Master Corporate Schema - FIXED ORDER RESILIENT
OFFICIAL_SCHEMA_ORDER = [
    "Document Type", 
    "Full Name", 
    "Arabic Name", 
    "Nationality",
    "Emirates ID Number (784-xxxx-xxxxxxx-x)", 
    "Passport Number",
    "Work Permit Number (9 Digits)", 
    "Personal Number (14 Digits)",
    "Permit / Security Pass / Card Number",
    "UID / Unified Number",
    "Date of Birth", 
    "Date of Issue", 
    "Date of Expiry", 
    "Employer / Company Name",
    "Occupation / Trade", 
    "Site Location", 
    "Arabic Site Location Names",
    "English Translation of Arabic Site Locations", 
    "Issuing Authority",
    "Any other visible document-specific fields", 
    "Confidence level for each extracted field",
    "Remarks for unclear or doubtful fields"
]

TARGET_MODEL = "models/gemini-2.5-flash"

st.title("📂 AI-Powered Document Data Extraction Tool")
st.subheader("Official Schema Compliant & Parallel Processing")

with st.sidebar:
    st.header("⚙️ Execution Settings")
    api_mode = st.selectbox("Execution Mode", ["Demo Mode (Simulated AI)", "Live Production Mode"])
    pipeline_speed = st.selectbox("Pipeline Speed", ["Parallel (Fast)", "Sequential (Safe)"])
    
    gemini_key = ""
    if api_mode == "Live Production Mode":
        gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")

    custom_field = st.text_input("➕ Extract Custom Field", key="custom_field_input")
    enable_consolidation = st.checkbox("Enable Person-Based Consolidation")

active_schema = OFFICIAL_SCHEMA_ORDER.copy()
if custom_field:
    active_schema.insert(19, custom_field)

col_upload, col_clear = st.columns([6, 1])
with col_upload:
    uploaded_files = st.file_uploader("Click to Add Documents", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True, key=f"uploader_{st.session_state['uploader_key']}")
with col_clear:
    if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
        st.session_state["uploader_key"] += 1 
        st.rerun()

def optimize_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
    max_size = 2400 
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=98)
    return buffer.getvalue()

def process_file_backend(file_bytes, unique_filename, incoming_key, model_name, target_schema):
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
            model=model_name,
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
        fail_row = {col: "-" for col in target_schema}
        fail_row["Source_File_Name"] = unique_filename
        fail_row["Document Type"] = "FAILED"
        fail_row["Remarks for unclear or doubtful fields"] = str(e)
        return fail_row

if st.button("🚀 Run Extraction Pipeline", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one document.")
    else:
        raw_results = []
        total_files = len(uploaded_files)
        progress_bar = st.progress(0)
        table_placeholder = st.empty() 
        
        unique_file_tuples = [(f.read(), f.name) for f in uploaded_files]
        
        if api_mode == "Live Production Mode" and pipeline_speed == "Parallel (Fast)":
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                futures = {executor.submit(process_file_backend, f_bytes, f_name, gemini_key, TARGET_MODEL, active_schema): f_name for f_bytes, f_name in unique_file_tuples}
                for future in concurrent.futures.as_completed(futures):
                    raw_results.append(future.result())
                    progress_bar.progress(len(raw_results) / total_files)
                    table_placeholder.dataframe(pd.DataFrame(raw_results), use_container_width=True)
        else:
            for f_bytes, f_name in unique_file_tuples:
                if api_mode == "Live Production Mode":
                    res = process_file_backend(f_bytes, f_name, gemini_key, TARGET_MODEL, active_schema)
                else:
                    res = {col: "-" for col in active_schema}
                    res["Source_File_Name"] = f_name
                raw_results.append(res)
                progress_bar.progress(len(raw_results) / total_files)
                table_placeholder.dataframe(pd.DataFrame(raw_results), use_container_width=True)

        st.success("🎉 İşlem tamamlandı!")
        
        if raw_results:
            df = pd.DataFrame(raw_results)
            if enable_consolidation and "Full Name" in df.columns:
                df = df.groupby('Full Name').agg(lambda x: ' / '.join(set(x.astype(str).str.strip()))).reset_index()
            st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "Report.csv", "text/csv")
