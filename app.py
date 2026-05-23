import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import json
import base64
import concurrent.futures
import time
from PIL import Image
import io

# Page Configuration
st.set_page_config(page_title="AI Powered Document Extraction Tool", layout="wide")

# Initialize Session State for File Uploader Key to enable global clear
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# Master Corporate Schema - FIXED ORDER RESILIENT (Do not modify order)
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

# Locked specifically onto Gemini 2.5 Flash
TARGET_MODEL = "models/gemini-2.5-flash"

# Main UI Headers
st.title("📂 AI-Powered Document Data Extraction Tool")
st.subheader("Official Schema Compliant & Parallel Processing")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Execution Settings")
    api_mode = st.selectbox("Execution Mode", ["Demo Mode (Simulated AI)", "Live Production Mode"])
    pipeline_speed = st.selectbox("Pipeline Speed", ["Parallel (Fast)", "Sequential (Safe)"])
    
    gemini_key = ""
    if api_mode == "Live Production Mode":
        gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
        st.caption("Please enter your Pay-as-you-go API key to remove rate limits.")

    st.write("---")
    st.subheader("➕ Extract Custom Field")
    custom_field = st.text_input("e.g., Sex, Blood Group", key="custom_field_input")
    
    st.write("---")
    enable_consolidation = st.checkbox("Enable Person-Based Consolidation")

# Dynamically update the target schema
active_schema = OFFICIAL_SCHEMA_ORDER.copy()
if custom_field:
    active_schema.insert(19, custom_field)

# --- FILE UPLOADER & BULK DELETE AREA ---
col_upload, col_clear = st.columns([6, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        "Click to Add Documents (You can select multiple files)", 
        type=["png", "jpg", "jpeg", "pdf"], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}"
    )

with col_clear:
    st.write(" ") 
    st.write(" ") 
    if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
        st.session_state["uploader_key"] += 1 
        st.rerun()

# Image Optimization Engine - High-Res Layout Preserved
def optimize_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    max_size = 2400 
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()

# Single File Processing Backend Motor
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
        
        # STRICT ZERO-HALLUCINATION & MULTI-DOCUMENT PROMPT (WITH ADVANCED ENTRY/RESIDENCE VISA LOGIC)
        schema_prompt = f"""You are an advanced corporate OCR engine with zero-tolerance for data misplacement, digit hallucination, or guessing.
        
        STEP 1: Identify 'Document Type' (UAE Emirates ID, UAE Labor Card / Work Permit, Security Pass, Passport, UAE Residence Permit / Visa).
        
        STEP 2: Map the variables strictly based on the identified document type into the requested keys:
        - If UAE Emirates ID: Map the 15-digit number to 'Emirates ID Number (784-xxxx-xxxxxxx-x)'.
        - If UAE Labor Card: Map 'Work Permit No' to 'Work Permit Number (9 Digits)' and 'Personal No' to 'Personal Number (14 Digits)'.
        - If Passport: Map passport serial to 'Passport Number'.
        - If Security Pass: Map badge/permit string to 'Permit / Security Pass / Card Number'.
        - If UAE Residence Permit / Visa: Map 'ID Number' to 'Emirates ID Number (784-xxxx-xxxxxxx-x)', 'Passport No' to 'Passport Number', and 'File / الملف' to 'UID / Unified Number'.
        
        CRITICAL ABSOLUTE RULES FOR ALL FIELDS (ZERO HALLUCINATION):
        1. DO NOT guess, alter, interpolate, or hallucinate any digits, letters, or characters.
        2. You must transcribe numbers and fields EXACTLY as they are visually printed.
        3. TRANSLATION RULE: You must translate 'Nationality' and 'Occupation / Trade' fields into official English (e.g., Change 'الهند' to 'India', 'نجار' to 'Carpenter', 'عامل' to 'Laborer').
        4. SPECIAL INTENSE RULE FOR UAE RESIDENCE/VISA NATIONALITY & ARABIC NAME: On UAE Residence Permit / Visa documents, look closely at the text strings at the bottom of the page, below the Expiry Date. The nationality country is written in Arabic but it is often concatenated or slightly typo-ridden inside lines like 'الجنسية الهندا' or 'والجنسية الهندا'. You MUST perform substring and sub-word checking: if you find 'الهند' or 'الهندا' anywhere in that bottom area text blocks, the Nationality is strictly 'India'. If you find 'باكستان', the Nationality is strictly 'Pakistan'. Translate this successfully and output it to the 'Nationality' field. Do not output '-' or 'Unreadable' for nationality if these character blocks are present.
        5. SPECIAL CHECK FOR UAE LABOR CARDS: The 'Work Permit Number' must ALWAYS be double-checked. If you see or tend to output the number starting with '126' or matching '126572649' on Gandharv Singh's card, verify it with the absolute raw text pixels. If it is blurry or has glare, DO NOT guess it. You MUST output 'Unreadable' instead of a hallucinated value.
        6. IF A FIELD OR DIGIT IS BLURRY, CORRUPTED, OBSCURED, OR UNREADABLE, and you are not 100% confident, DO NOT invent or guess a value. You MUST set that field's value to "Unreadable" and explain why in 'Remarks for unclear or doubtful fields'.
        7. Output must populate exactly these keys: {json.dumps(target_schema)}.
        8. If a field completely does not apply to the document type, set its value to "-"."""
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                schema_prompt,
                {"inline_data": {"data": base64_data, "mime_type": mime_type}}
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        extracted_json = json.loads(response.text.strip())
        
        safe_json = {"Source_File_Name": unique_filename}
        for col in target_schema:
            found_key = next((k for k in extracted_json if k.lower().strip() == col.lower().strip()), None)
            safe_json[col] = extracted_json[found_key] if found_key else "-"
            
        return safe_json
    except Exception as e:
        error_msg = str(e)
        status_label = "FAILED / RATE LIMIT EXCEEDED" if "resource" in error_msg.lower() or "limit" in error_msg.lower() else f"FAILED / ERROR"
        
        fail_row = {col: "-" for col in target_schema}
        fail_row["Source_File_Name"] = unique_filename
        fail_row["Document Type"] = status_label
        fail_row["Remarks for unclear or doubtful fields"] = f"Technical Details: {error_msg}"
        return fail_row

# Execution Pipeline Trigger Button
if st.button("🚀 Run Extraction Pipeline", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one document.")
    elif api_mode == "Live Production Mode" and not gemini_key:
        st.error("Please enter your Gemini API Key for Production Mode.")
    else:
        raw_results = []
        total_files = len(uploaded_files)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        table_placeholder = st.empty() 
        
        BATCH_SIZE = 5
        
        # Thread-Safe In-Memory Document Structuring
        name_counts = {}
        unique_file_tuples = []
        for f in uploaded_files:
            orig_name = f.name
            in_memory_bytes = f.read()
            
            if orig_name not in name_counts:
                name_counts[orig_name] = 1
                unique_name = orig_name
            else:
                name_counts[orig_name] += 1
                name_parts = orig_name.rsplit('.', 1)
                if len(name_parts) == 2:
                    unique_name = f"{name_parts[0]}_{name_counts[orig_name]}.{name_parts[1]}"
                else:
                    unique_name = f"{orig_name}_{name_counts[orig_name]}"
            
            unique_file_tuples.append((in_memory_bytes, unique_name))
        
        # Batch Execution Loop
        for i in range(0, total_files, BATCH_SIZE):
            batch = unique_file_tuples[i:i+BATCH_SIZE]
            status_text.markdown(f"🔄 **Processing:** {i} / {total_files} documents processed. Injecting next batch...")
            
            batch_results = []
            
            if api_mode == "Live Production Mode":
                if pipeline_speed == "Parallel (Fast)":
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(process_file_backend, f_bytes, f_name, gemini_key, TARGET_MODEL, active_schema) for f_bytes, f_name in batch]
                        for future in concurrent.futures.as_completed(futures):
                            batch_results.append(future.result())
                else:
                    for f_bytes, f_name in batch:
                        res = process_file_backend(f_bytes, f_name, gemini_key, TARGET_MODEL, active_schema)
                        batch_results.append(res)
                        time.sleep(1) 
            else:
                for f_bytes, f_name in batch:
                    mock_row = {col: "-" for col in active_schema}
                    mock_row["Source_File_Name"] = f_name
                    mock_row["Document Type"] = "Simulated Residence Visa"
                    mock_row["Full Name"] = "PIYUSH SHUKLA RAMGOPAL SHUKLA"
                    mock_row["Nationality"] = "India"
                    mock_row["Confidence level for each extracted field"] = "High"
                    mock_row["Remarks for unclear or doubtful fields"] = "-"
                    batch_results.append(mock_row)
            
            raw_results.extend(batch_results)
            
            current_progress = min((i + BATCH_SIZE) / total_files, 1.0)
            progress_bar.progress(current_progress)
            
            # Live Preview Table Rendering (Shows failures immediately)
            if raw_results:
                current_df = pd.DataFrame(raw_results)
                cols_order = ["Source_File_Name"] + [c for c in current_df.columns if c != "Source_File_Name"]
                current_df = current_df[cols_order]
                table_placeholder.dataframe(current_df, use_container_width=True)

        status_text.success("🎉 All documents processed successfully at maximum speed!")
        progress_bar.empty()

        if raw_results:
            df = pd.DataFrame(raw_results)
            cols = ["Source_File_Name"] + [c for c in df.columns if c != "Source_File_Name"]
            df = df[cols]
            
            for r in raw_results:
                if "FAILED" in str(r.get("Document Type", "")):
                    st.warning(f"⚠️ {r['Source_File_Name']} could not be parsed by the AI. Marked in the report spreadsheet.")
            
            # --- PERSON-BASED CONSOLIDATION ENGINE ---
            if enable_consolidation and "Full Name" in df.columns:
                df['Full Name'] = df['Full Name'].astype(str).str.strip().str.upper()
                df = df.groupby('Full Name').agg(lambda x: ' / '.join(set(x.astype(str).str.strip()))).reset_index()
                df = df[cols]
            
            table_placeholder.dataframe(df, use_container_width=True)
            
            @st.cache_data
            def convert_df(df_to_download):
                return df_to_download.to_csv(index=False).encode('utf-8')
                
            csv_bytes = convert_df(df)
            st.download_button(
                label="📥 Download Consolidated Corporate Report (CSV)",
                data=csv_bytes,
                file_name="Consolidated_AI_Document_Report.csv",
                mime="text/csv"
            )
