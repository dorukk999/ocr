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

# Görüntü Boyutunu Küçülterek Sistemi Hafifleten Motor
def optimize_image(file):
    img = Image.open(file)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    max_size = 1600
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()

# Single File Processing Backend Motor
def process_file_backend(file, incoming_key, model_name):
    try:
        client = genai.Client(api_key=incoming_key)
        optimized_bytes = optimize_image(file)
        base64_data = base64.b64encode(optimized_bytes).decode("utf-8")
        
        schema_prompt = f"""You are an advanced corporate OCR engine with zero-tolerance for digit hallucination.
        STEP 1: Identify 'Document Type' (UAE Emirates ID, UAE Labor Card / Work Permit, Security Pass, Passport).
        STEP 2: Map variables strictly:
        - Emirates ID -> 'Emirates ID Number (784-xxxx-xxxxxxx-x)' (15 digits)
        - Labor Card -> 'Work Permit Number (9 Digits)' and 'Personal Number (14 Digits)'
        - Passport -> 'Passport Number'
        - Security Pass -> 'Permit / Security Pass / Card Number'
        - UID/Unified ID -> 'UID / Unified Number'
        
        CRITICAL ABSOLUTE RULES FOR ALL FIELDS AND DOCUMENTS:
        1. DO NOT guess, alter, interpolate, or hallucinate any digits, letters, or characters.
        2. Transcribe numbers EXACTLY as they are visually printed on the card.
        3. If a character is blurry/corrupted, document doubt in 'Remarks for unclear or doubtful fields'. Never invent values.
        4. Output must populate exactly these keys: {json.dumps(active_schema)}.
        5. If a field doesn't apply, explicitly set value to "-"."""
        
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
        safe_json = {"Source_File_Name": file.name}
        for col in active_schema:
            found_key = next((k for k in extracted_json if k.lower().strip() == col.lower().strip()), None)
            safe_json[col] = extracted_json[found_key] if found_key else "-"
            
        return safe_json
    except Exception as e:
        return {"Source_File_Name": file.name, "Error": str(e)}

# Execution Pipeline Trigger Button
if st.button("🚀 Run Extraction Pipeline", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one document.")
    elif api_mode == "Live Production Mode" and not gemini_key:
        st.error("Please enter your Gemini API Key for Production Mode.")
    else:
        raw_results = []
        total_files = len(uploaded_files)
        
        # Ekranın donmasını engellemek için canlı durum alanları oluşturuyoruz
        progress_bar = st.progress(0)
        status_text = st.empty()
        table_placeholder = st.empty() # Canlı tablo önizleme alanı
        
        # BATCH SIZE: Ücretsiz katmanda donmayı önleyecek en ideal paket boyutu (5'er döküman)
        BATCH_SIZE = 5
        
        # Dosyaları paketlere bölüyoruz
        for i in range(0, total_files, BATCH_SIZE):
            batch = uploaded_files[i:i+BATCH_SIZE]
            status_text.markdown(f"🔄 **Sistem Yükü Dengeleniyor:** {i} / {total_files} döküman tamamlandı. Yeni paket işleniyor...")
            
            batch_results = []
            
            if api_mode == "Live Production Mode":
                if pipeline_speed == "Parallel (Fast)":
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(process_file_backend, f, gemini_key, TARGET_MODEL) for f in batch]
                        for future in concurrent.futures.as_completed(futures):
                            batch_results.append(future.result())
                else:
                    for f in batch:
                        res = process_file_backend(f, gemini_key, TARGET_MODEL)
                        batch_results.append(res)
                        time.sleep(1) # Paket içi hafif bekleme
            else:
                for f in batch:
                    mock_row = {col: "-" for col in active_schema}
                    mock_row["Source_File_Name"] = f.name
                    mock_row["Document Type"] = "Simulated ID Card"
                    mock_row["Full Name"] = "ADITYA UPENDRA GANDHI"
                    mock_row["Nationality"] = "India"
                    mock_row["Confidence level for each extracted field"] = "High"
                    mock_row["Remarks for unclear or doubtful fields"] = "-"
                    batch_results.append(mock_row)
            
            raw_results.extend(batch_results)
            
            # İlerleme çubuğunu güncelle
            current_progress = min((i + BATCH_SIZE) / total_files, 1.0)
            progress_bar.progress(current_progress)
            
            # CANLI TABLO GÜNCELLEME: Müşteri her 5 dökümanda bir tablonun büyüdüğünü görecek
            valid_batch_data = [r for r in raw_results if "Error" not in r]
            if valid_batch_data:
                current_df = pd.DataFrame(valid_batch_data)
                cols_order = ["Source_File_Name"] + [c for c in current_df.columns if c != "Source_File_Name"]
                current_df = current_df[cols_order]
                table_placeholder.dataframe(current_df, use_container_width=True)
            
            # Paketler arası Google Free Tier Kota Koruması (Bloke olmayı önleyen altın kural)
            if api_mode == "Live Production Mode" and (i + BATCH_SIZE) < total_files:
                status_text.markdown("⏳ **Google Kota Koruması Devrede:** Sistem hız limiti (Rate Limit) yememek için 10 saniye dinleniyor...")
                time.sleep(10)

        status_text.success("🎉 Tüm dökümanlar başarıyla eritildi!")
        progress_bar.empty()

        # Filter out errors and map to final DataFrame
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
            
            # Final Tabloyu Sabitle
            table_placeholder.dataframe(df, use_container_width=True)
            
            # Download Corporate CSV Data Report Document
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
