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

# Master Corporate Schema - Multi-Document Resilient
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

# File Uploader Asset
uploaded_files = st.file_uploader("Click to Add Documents (You can select multiple files)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Görüntü Boyutunu ve Kalitesini Optimize Eden Motor (Rakam güvenliği için %95 netlik)
def optimize_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    max_size = 1600
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    # CRITICAL UPDATE: Kalite %95'e çıkarıldı, ince yazılar korundu.
    img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()

# Single File Processing Backend Motor - Refactored for Thread-Safe Bytes
def process_file_backend(file_bytes, unique_filename, incoming_key, model_name, target_schema):
    try:
        client = genai.Client(api_key=incoming_key)
        
        # Optimize edilmiş yüksek netlikteki byte verisi kullanımı
        optimized_bytes = optimize_image(file_bytes)
        base64_data = base64.b64encode(optimized_bytes).decode("utf-8")
        
        # SÜPER KATI - SIFIR HALÜSİNASYON PROMPT YAPISI
        schema_prompt = f"""You are an advanced corporate OCR engine with zero-tolerance for data misplacement, digit hallucination, or guessing.
        
        STEP 1: Identify 'Document Type' (UAE Emirates ID, UAE Labor Card / Work Permit, Security Pass, Passport).
        
        STEP 2: Based on the identified document type, extract the parameters under strict rules:
        - If the document is an Emirates ID, extract the 15-digit number into 'Emirates ID Number (784-xxxx-xxxxxxx-x)' (15 digits)
        - If the document is a Labor Card / Work Permit, extract the 9-digit Work Permit No into 'Work Permit Number (9 Digits)' and the 14-digit Personal No into 'Personal Number (14 Digits)'
        - If the document is a Passport, extract the passport serial string into 'Passport Number'
        - If it is a Site/Security/Access pass, map its badge or permit number into 'Permit / Security Pass / Card Number'
        - If a 'UID' or 'Unified ID' is visible anywhere on the document, map it to 'UID / Unified Number'
        
        CRITICAL ABSOLUTE RULES FOR ALL FIELDS (ZERO HALLUCINATION):
        1. DO NOT guess, alter, interpolate, or hallucinate any digits, letters, or characters.
        2. You must transcribe numbers and fields EXACTLY as they are visually printed.
        3. TRANSLATION RULE: You must translate 'Nationality' and 'Occupation / Trade' fields into official English (e.g., Change 'الهند' to 'India', 'نجار' to 'Carpenter', 'عامل' to 'Laborer').
        4. IF A FIELD OR DIGIT IS BLURRY, CORRUPTED, OBSCURED, OR UNREADABLE, and you are not 100% confident, DO NOT invent or guess a value. You MUST set that field's value to "Okunamadı / Unreadable" and explain why in 'Remarks for unclear or doubtful fields'.
        5. Output must populate exactly these keys: {json.dumps(target_schema)}.
        6. If a field completely does not apply to the document type, set its value to "-"."""
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                schema_prompt,
                {"inline_data": {"data": base64_data, "mime_type": "image/jpeg"}}
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
        return {"Source_File_Name": unique_filename, "Error": str(e)}

# Execution Pipeline Trigger Button
if st.button("🚀 Run Extraction Pipeline", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one döküman.")
    elif api_mode == "Live Production Mode" and not gemini_key:
        st.error("Please enter your Gemini API Key for Production Mode.")
    else:
        raw_results = []
        total_files = len(uploaded_files)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        table_placeholder = st.empty() 
        
        BATCH_SIZE = 5
        
        # --- DOSYALARI ÖNCEDEN BELLEĞE ALAN VE BENZERSİZLEŞTİREN GÜVENLİ MOTOR ---
        name_counts = {}
        unique_file_tuples = []
        for f in uploaded_files:
            orig_name = f.name
            
            # Streamlit I/O hatasını önlemek için içeriği erkenden RAM'e kopyalıyoruz
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
        
        # Paketler halinde döngüyü çalıştır
        for i in range(0, total_files, BATCH_SIZE):
            batch = unique_file_tuples[i:i+BATCH_SIZE]
            status_text.markdown(f"🔄 **Sistem Yükü Dengeleniyor:** {i} / {total_files} döküman tamamlandı. Yeni paket işleniyor...")
            
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
                    mock_row["Document Type"] = "Simulated ID Card"
                    mock_row["Full Name"] = "ADITYA UPENDRA GANDHI"
                    mock_row["Nationality"] = "India"
                    mock_row["Confidence level for each extracted field"] = "High"
                    mock_row["Remarks for unclear or doubtful fields"] = "-"
                    batch_results.append(mock_row)
            
            raw_results.extend(batch_results)
            
            current_progress = min((i + BATCH_SIZE) / total_files, 1.0)
            progress_bar.progress(current_progress)
            
            # Canlı önizleme tablosu güncellemesi
            valid_batch_data = [r for r in raw_results if "Error" not in r]
            if valid_batch_data:
                current_df = pd.DataFrame(valid_batch_data)
                cols_order = ["Source_File_Name"] + [c for c in current_df.columns if c != "Source_File_Name"]
                current_df = current_df[cols_order]
                table_placeholder.dataframe(current_df, use_container_width=True)
            
            # KOTA KORUMASI BEKLEME SÜRESİ (Ücretsiz katman testlerin için 35 saniye)
            if api_mode == "Live Production Mode" and (i + BATCH_SIZE) < total_files:
                status_text.markdown("⏳ **Google Kota Koruması Devrede:** Ücretsiz API hız limiti yememek için sistem **35 saniye** dinleniyor...")
                time.sleep(35)

        status_text.success("🎉 Tüm dökümanlar başarıyla eritildi!")
        progress_bar.empty()

        final_data = []
        for r in raw_results:
            if "Error" in r:
                st.warning(f"⚠️ {r['Source_File_Name']} işlenirken hata oluştu: {r['Error']}")
            else:
                final_data.append(r)
                
        if final_data:
            df = pd.DataFrame(final_data)
            cols = ["Source_File_Name"] + [c for c in df.columns if c != "Source_File_Name"]
            df = df[cols]
            
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
