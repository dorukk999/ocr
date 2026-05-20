import streamlit as st
import pandas as pd
from google import genai
import json
import base64
import concurrent.futures

# Sayfa Genişlik Ayarları
st.set_page_config(page_title="AI Powered Document Extraction Tool", layout="wide")

# Şema Yapısı
OFFICIAL_SCHEMA_ORDER = [
    "Document Type", "Full Name", "Arabic Name", "Nationality",
    "Emirates ID / Document Number", "Passport Number", "Date of Birth",
    "Date of Issue", "Date of Expiry", "Employer / Company Name",
    "Occupation / Trade", "Site Location", "Arabic Site Location Names",
    "English Translation of Arabic Site Locations", "Issuing Authority",
    "Any other visible document-specific fields", "Confidence level for each extracted field",
    "Remarks for unclear or doubtful fields"
]

# Başlık
st.title("📂 AI-Powered Document Data Extraction Tool")
st.subheader("Official Schema Compliant & Parallel Processing")

# Sol Panel (Sidebar) - Ayarlar
with st.sidebar:
    st.header("⚙️ Execution Settings")
    
    api_mode = st.selectbox("Execution Mode", ["Demo Mode (Simulated AI)", "Live Production Mode"])
    
    # Model Seçimi
    ai_model = st.selectbox("AI Model", [
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro"
    ])
    
    # Hız Ayarı
    pipeline_speed = st.selectbox("Pipeline Speed", ["Parallel (Fast)", "Sequential (Safe)"])
    
    # API Key Alanı
    gemini_key = ""
    if api_mode == "Live Production Mode":
        gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
        st.caption("Katman limitlerini kaldırmak için Pay-as-you-go anahtarınızı girin.")

    # Dinamik Alan Ekleme
    st.write("---")
    st.subheader("➕ Extract Custom Field")
    custom_field = st.text_input("e.g., Sex, Blood Group", key="custom_field_input")
    
    # Kişi Bazlı Konsolidasyon
    st.write("---")
    enable_consolidation = st.checkbox("Enable Person-Based Consolidation")

# Dinamik şemayı güncelle
active_schema = OFFICIAL_SCHEMA_ORDER.copy()
if custom_field:
    active_schema.insert(15, custom_field)

# Dosya Yükleme Alanı
uploaded_files = st.file_uploader("Click to Add Documents (You can select multiple files)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Tekil Dosya İşleme Fonksiyonu (Python Backend)
def process_file_backend(file, key, model_name):
    try:
        client = genai.Client(api_key=key)
        
        # Dosyayı oku ve Base64'e çevir
        file_bytes = file.read()
        base64_data = base64.b64encode(file_bytes).decode("utf-8")
        
        schema_prompt = f"""You are an expert corporate OCR text extraction engine. 
        Analyze the document image and extract all text parameters precisely. 
        CRITICAL: Return a single flat JSON object where keys EXACTLY match these names (case-sensitive): {json.dumps(active_schema)}. 
        If any specific field data is missing from the image, explicitly set its value to "-". Do not omit any key."""
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                schema_prompt,
                {"inline_data": {"data": base64_data, "mime_type": "image/jpeg"}}
            ]
        )
        
        # Gelen yanıtı temizle ve JSON'a çevir
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        extracted_json = json.loads(clean_text)
        
        # Şemaya tam uyumluluk sağla
        safe_json = {"Source_File_Name": file.name}
        for col in active_schema:
            found_key = next((k for k in extracted_json if k.lower().strip() == col.lower().strip()), None)
            safe_json[col] = extracted_json[found_key] if found_key else "-"
            
        return safe_json
    except Exception as e:
        return {"Source_File_Name": file.name, "Error": str(e)}

# Çalıştırma Butonu
if st.button("🚀 Run Extraction Pipeline", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one document.")
    elif api_mode == "Live Production Mode" and not gemini_key:
        st.error("Please enter your Gemini API Key for Production Mode.")
    else:
        raw_results = []
        
        with st.spinner("Processing documents... Please wait."):
            # --- MOD 1: CANLI ÜRETİM (PRODUCTION) ---
            if api_mode == "Live Production Mode":
                if pipeline_speed == "Parallel (Fast)":
                    # Eşzamanlı Paralel İşleme (Maksimum Hız)
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(process_file_backend, f, gemini_key, ai_model) for f in uploaded_files]
                        for future in concurrent.futures.as_completed(futures):
                            raw_results.append(future.result())
                else:
                    # Sıralı İşleme
                    for f in uploaded_files:
                        res = process_file_backend(f, gemini_key, ai_model)
                        raw_results.append(res)
            
            # --- MOD 2: DEMO MODU (SIMULATION) ---
            else:
                for f in uploaded_files:
                    mock_row = {col: "-" for col in active_schema}
                    mock_row["Source_File_Name"] = f.name
                    mock_row["Document Type"] = "Simulated ID Card"
                    mock_row["Full Name"] = "ADITYA UPENDRA GANDHI"
                    mock_row["Nationality"] = "India"
                    mock_row["Confidence level for each extracted field"] = "High"
                    raw_results.append(mock_row)

        # Hataları Ayıkla ve Veriyi DataFrame'e Çevir
        final_data = []
        for r in raw_results:
            if "Error" in r:
                st.warning(f"⚠️ {r['Source_File_Name']} could not be processed: {r['Error']}")
            else:
                final_data.append(r)
                
        if final_data:
            df = pd.DataFrame(final_data)
            
            # Sütun sıralamasını düzelt (Source_File_Name her zaman ilk sütun olsun)
            cols = ["Source_File_Name"] + [c for c in df.columns if c != "Source_File_Name"]
            df = df[cols]
            
            # --- KONSOLİDASYON FİLTRESİ ---
            if enable_consolidation and "Full Name" in df.columns:
                df['Full Name'] = df['Full Name'].astype(str).str.strip().str.upper()
                # Aynı isimdeki satırları birleştir
                df = df.groupby('Full Name').agg(lambda x: ' / '.join(set(x.astype(str).str.strip()))).reset_index()
                # Kolon sıralamasını koru
                df = df[cols]
            
            # Veri Önizleme Tablosu
            st.success(f"Successfully processed {len(df)} profiles!")
            st.dataframe(df, use_container_width=True)
            
            # Excel/CSV İndirme Butonu
            @st.cache_data
            def convert_df(df_to_download):
                return df_to_download.to_csv(index=False).encode('utf-8')
                
            csv_bytes = convert_df(df)
            st.download_button(
                label="📥 Download Consolidated Corporate Report (CSV)",
                data=csv_bytes,
                file_name="Musteri_Konsolide_Rapor.csv",
                mime="text/csv"
            )
