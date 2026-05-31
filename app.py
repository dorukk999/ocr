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

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

OFFICIAL_SCHEMA_ORDER = [
    "Document Type", "Full Name", "Arabic Name", "Nationality",
    "Emirates ID Number (784-xxxx-xxxxxxx-x)", "Passport Number",
    "Work Permit Number (9 Digits)", "Personal Number (14 Digits)",
    "Permit / Security Pass / Card Number", "UID / Unified Number",
    "Date of Birth", "Date of Issue", "Date of Expiry", "Employer / Company Name",
    "Occupation / Trade", "Site Location", "Arabic Site Location Names",
    "English Translation of Arabic Site Locations", "Issuing Authority",
    "Any other visible document-specific fields", "Confidence level for each extracted field",
    "Remarks for unclear or doubtful fields"
]

TARGET_MODEL = "models/gemini-2.0-flash"

st.title("📂 AI-Powered Document Data Extraction Tool")
st.subheader("Official Schema Compliant & Parallel Processing")

with st.sidebar:
    st.header("⚙️ Execution Settings")
    api_mode = st.selectbox("Execution Mode", ["Demo Mode (Simulated AI)", "Live Production Mode"])
    pipeline_speed = st.selectbox("Pipeline Speed", ["Parallel (Fast)", "Sequential (Safe)"])
    gemini_key = st.text_input("Gemini API Key", type="password") if api_mode == "Live Production Mode" else ""
    custom_field = st.text_input("➕ Extract Custom Field")
    enable_consolidation = st.checkbox("Enable Person-Based Consolidation")

active_schema = OFFICIAL_SCHEMA_ORDER.copy()
if custom_field:
    active_schema.insert(19, custom_field)

def optimize_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
    max_size = 2400 
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=98)
    return buffer.getvalue()

# OPTIMIZED: Client parametresi dışarıdan alınıyor (bağlantı havuzu)
def process_file_backend(file_bytes, unique_filename, client, model_name, target_schema):
    try:
        is_pdf = unique_filename.lower().endswith('.pdf')
        base64_data = base64.b64encode(file_bytes if is_pdf else optimize_image(file_bytes)).decode("utf-8")
        mime_type = "application/pdf" if is_pdf else "image/jpeg"
        
        schema_prompt = f"You are a strict corporate OCR. Output exactly in JSON format using these keys: {json.dumps(target_schema)}."
        
        response = client.models.generate_content(
            model=model_name,
            contents=[schema_prompt, {"inline_data": {"data": base64_data, "mime_type": mime_type}}],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        extracted_json = json.loads(response.text.strip())
        safe_json = {"Source_File_Name": unique_filename}
        for col in target_schema:
            safe_json[col] = extracted_json.get(col, "-")
        return safe_json
    except Exception as e:
        return {"Source_File_Name": unique_filename, "Document Type": "FAILED", "Remarks for unclear or doubtful fields": str(e)}

col_upload, col_clear = st.columns([6, 1])
with col_upload:
    uploaded_files = st.file_uploader("Add Documents", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True, key=f"uploader_{st.session_state['uploader_key']}")
with col_clear:
    if st.button("🗑️ Clear"): st.session_state["uploader_key"] += 1; st.rerun()

if st.button("🚀 Run Extraction Pipeline", type="primary"):
    if not uploaded_files: st.error("No files."); st.stop()
    
    raw_results = []
    unique_file_tuples = [(f.read(), f.name) for f in uploaded_files]
    
    # Hız Optimizasyonu: Parallel ve Max Workers = 10
    if api_mode == "Live Production Mode":
        client = genai.Client(api_key=gemini_key)
        if pipeline_speed == "Parallel (Fast)":
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_file_backend, f_bytes, f_name, client, TARGET_MODEL, active_schema) for f_bytes, f_name in unique_file_tuples]
                raw_results = [f.result() for f in concurrent.futures.as_completed(futures)]
        else:
            for f_bytes, f_name in unique_file_tuples:
                raw_results.append(process_file_backend(f_bytes, f_name, client, TARGET_MODEL, active_schema))
    else:
        raw_results = [{"Source_File_Name": f[1], "Document Type": "Simulated", "Full Name": "Demo User"} for f in unique_file_tuples]

    df = pd.DataFrame(raw_results)
    if enable_consolidation and "Full Name" in df.columns:
        df = df.groupby('Full Name').agg(lambda x: ' / '.join(set(x.astype(str)))).reset_index()
    
    st.dataframe(df, use_container_width=True)
    st.download_button("📥 Download CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="Report.csv", mime="text/csv")
