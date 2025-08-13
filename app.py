import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- KONSTANTA GLOBAL ---
PRIMARY_COLOR = "#2C2F7F"
ACCENT_COLOR = "#7AA02F"
BACKGROUND_COLOR = "#EAF0FA"
TEXT_COLOR = "#26272E"
HEADER_BACKGROUND_COLOR = ACCENT_COLOR
SIDEBAR_HIGHLIGHT_COLOR = "#4A5BAA"
ACTIVE_BUTTON_BG_COLOR = "#3F51B5"
ACTIVE_BUTTON_TEXT_COLOR = "#FFFFFF"
ACTIVE_BUTTON_BORDER_COLOR = "#FFD700"

ID_COLS = ["No", "Nama", "JK", "Kelas"]
NUMERIC_COLS = ["Rata Rata Nilai Akademik", "Kehadiran"]
CATEGORICAL_COLS = ["Ekstrakurikuler Komputer", "Ekstrakurikuler Pertanian",
                    "Ekstrakurikuler Menjahit", "Ekstrakurikuler Pramuka"]
ALL_FEATURES_FOR_CLUSTERING = NUMERIC_COLS + CATEGORICAL_COLS

# --- CUSTOM CSS & HEADER ---
custom_css = f"""
<style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }}
    .main .block-container {{
        padding-top: 7.5rem;
        padding-right: 4rem;
        padding-left: 4rem;
        padding-bottom: 3rem;
        max-width: 1200px;
        margin: auto;
    }}
    [data-testid="stVerticalBlock"] > div:not(:last-child),
    [data-testid="stHorizontalBlock"] > div:not(:last-child) {{
        margin-bottom: 0.5rem !important;
        padding-bottom: 0px !important;
    }}
    .stVerticalBlock, .stHorizontalBlock {{
        gap: 1rem !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        color: {PRIMARY_COLOR};
        font-weight: 600;
    }}
    h1 {{ font-size: 2.5em; }}
    h2 {{ font-size: 2em; }}
    h3 {{ font-size: 1.5em; }}
    .stApp > div > div:first-child > div:nth-child(2) [data-testid="stText"] {{
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        font-size: 0.95em;
        color: #666666;
    }}
    .stApp > div > div:first-child > div:nth-child(3) h1:first-child,
    .stApp > div > div:first-child > div:nth-child(3) h2:first-child,
    .stApp > div > div:first-child > div:nth-child(3) h3:first-child
    {{
        margin-top: 1rem !important;
    }}
    .stApp > div > div:first-child > div:nth-child(3) [data-testid="stAlert"]:first-child {{
        margin-top: 1.2rem !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {PRIMARY_COLOR};
        color: #ffffff;
        padding-top: 2.5rem;
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background-color: {PRIMARY_COLOR} !important;
        color: white !important;
        border: none !important;
        padding: 12px 25px !important;
        text-align: left !important;
        width: 100% !important;
        font-size: 17px !important;
        font-weight: 500 !important;
        margin: 0 !important;
        border-radius: 0 !important;
        transition: background-color 0.2s, color 0.2s, border-left 0.2s, box-shadow 0.2s;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center;
        gap: 10px;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {SIDEBAR_HIGHLIGHT_COLOR} !important;
        color: #e0e0e0 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] {{
        margin-bottom: 0px !important;
        padding: 0px !important;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
        margin-bottom: 0px !important;
    }}
    [data-testid="stSidebar"] .st-sidebar-button-active {{
        background-color: {ACTIVE_BUTTON_BG_COLOR} !important;
        color: {ACTIVE_BUTTON_TEXT_COLOR} !important;
        border-left: 6px solid {ACTIVE_BUTTON_BORDER_COLOR} !important;
        box-shadow: inset 4px 0 10px rgba(0,0,0,0.4) !important;
    }}
    [data-testid="stSidebar"] .st-sidebar-button-active > button {{
        background-color: {ACTIVE_BUTTON_BG_COLOR} !important;
        color: {ACTIVE_BUTTON_TEXT_COLOR} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stSidebar"] .stButton > button:not(.st-sidebar-button-active) {{
        border-left: 6px solid transparent !important;
        box-shadow: none !important;
    }}
    .custom-header {{
        background-color: {HEADER_BACKGROUND_COLOR};
        padding: 25px 40px;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.25);
        position: sticky;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        margin: 0 !important;
    }}
    .custom-header h1 {{
        margin: 0 !important;
        font-size: 32px;
        font-weight: bold;
        color: white;
    }}
    .custom-header .kanan {{
        font-weight: 600;
        font-size: 19px;
        color: white;
        opacity: 0.9;
        text-align: right;
    }}
    @media (max-width: 768px) {{
        .custom-header {{
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 20px;
            text-align: left;
        }}
        .custom-header h1 {{
            font-size: 24px;
            margin-bottom: 5px !important;
        }}
        .custom-header .kanan {{
            font-size: 14px;
            text-align: left;
        }}
        .main .block-container {{
            padding-top: 10rem;
            padding-right: 1rem;
            padding-left: 1rem;
        }}
    }}
    .stAlert {{
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px !important;
        margin-top: 20px !important;
        font-size: 0.95em;
        line-height: 1.5;
    }}
    .stAlert.info {{
        background-color: #e3f2fd;
        color: #1976D2;
        border-left: 6px solid #2196F3;
    }}
    .stAlert.success {{
        background-color: #e8f5e9;
        color: #388E3C;
        border-left: 6px solid #4CAF50;
    }}
    .stAlert.warning {{
        background-color: #fffde7;
        color: #FFA000;
        border-left: 6px solid #FFC107;
    }}
    .stAlert.error {{
        background-color: #ffebee;
        color: #D32F2F;
        border-left: 6px solid #F44336;
    }}
    .stForm {{
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 25px !important;
        margin-bottom: 25px !important;
        border: 1px solid #e0e0e0;
    }}
    .stDataFrame, .stTable {{
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-top: 30px !important;
        margin-bottom: 30px !important;
        border: 1px solid #e0e0e0;
    }}
    .stTable table th {{
        background-color: #f5f5f5 !important;
        color: {PRIMARY_COLOR} !important;
        font-weight: bold;
    }}
    .stTable table td {{
        padding: 8px 12px !important;
    }}
    .stButton > button {{
        background-color: {ACCENT_COLOR};
        color: white;
        padding: 10px 25px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
        margin-top: 15px !important;
        margin-bottom: 8px !important;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .stButton > button:hover {{
        background-color: {PRIMARY_COLOR};
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.25);
    }}
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }}
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {{
        border-radius: 8px;
        border: 1px solid #D1D1D1;
        padding: 10px 15px;
        margin-bottom: 8px !important;
        margin-top: 8px !important;
        background-color: white;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }}
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stCheckbox label, .stRadio label {{
        margin-bottom: 5px !important;
        padding-bottom: 0px !important;
        font-size: 0.98em;
        font-weight: 500;
        color: {TEXT_COLOR};
    }}
    div[data-testid="stSelectbox"] > div:first-child {{
        width: 480px;
        min-width: 300px;
    }}
    div[data-testid="stSelectbox"] > div > div > div > div[role="button"] {{
        width: 100% !important;
        white-space: normal;
        overflow: hidden;
        text-overflow: ellipsis;
        display: flex;
        align-items: center;
        height: auto;
        box-sizing: border-box;
        padding-right: 35px;
    }}
    div[role="listbox"][aria-orientation="vertical"] {{
        width: 500px !important;
        max-width: 600px !important;
        min-width: 400px !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        box-sizing: border-box;
        border-radius: 8px;
        border: 1px solid #D1D1D1;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: white;
    }}
    div[role="option"] {{
        white-space: normal !important;
        word-wrap: break-word !important;
        padding-right: 15px !important;
        padding-left: 15px !important;
        line-height: 1.4;
        min-height: 38px;
        display: flex;
        align-items: center;
    }}
    div[role="option"]:hover {{
        background-color: #e0e0e0;
        color: {PRIMARY_COLOR};
    }}
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {ACCENT_COLOR};
        border-radius: 5px;
    }}
    ::-webkit-scrollbar-track {{
        background: #e9e9e9;
    }}
    .stCheckbox label, .stRadio label {{
        display: flex;
        align-items: center;
        cursor: pointer;
        user-select: none;
    }}
    .stCheckbox {{
        margin-bottom: 10px !important;
        margin-top: 10px !important;
    }}
    .stExpander {{
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }}
    .stExpander > div > div > p {{
        font-weight: 600;
        color: {PRIMARY_COLOR};
    }}
    div[data-testid="column"] {{
        gap: 2rem;
    }}
    .stApp > div > div:first-child > div:nth-child(3) > div:first-child {{
        margin-top: 0rem !important;
    }}
    .login-container {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 80vh;
        text-align: center;
    }}
    .login-card {{
        background-color: white;
        padding: 50px 70px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        width: 100%;
        max-width: 600px;
        margin-top: 50px;
    }}
    .login-card h2 {{
        color: {PRIMARY_COLOR};
        font-size: 2.2em;
        margin-bottom: 2rem;
    }}
    .stButton > button {{
        background-color: {ACCENT_COLOR};
        color: white;
        padding: 10px 25px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
        margin-top: 15px !important;
        margin-bottom: 8px !important;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
</style>
"""

header_html = f"""
<div class="custom-header">
    <div><h1>PENGELOMPOKAN SISWA</h1></div>
    <div class="kanan">MADRASAH ALIYAH AL-HIKMAH</div>
</div>
"""

st.set_page_config(page_title="Klasterisasi K-Prototype Siswa", layout="wide", initial_sidebar_state="expanded")
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(header_html, unsafe_allow_html=True)

# --- FUNGSI PEMBANTU ---

def generate_pdf_profil_siswa(nama, data_siswa_dict, klaster, cluster_desc_map):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(44, 47, 127)
    pdf.cell(0, 10, "PROFIL SISWA - HASIL KLASTERISASI", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    keterangan_umum = (
        "Laporan ini menyajikan profil detail siswa berdasarkan hasil pengelompokan "
        "menggunakan Algoritma K-Prototype. Klasterisasi dilakukan berdasarkan "
        "nilai akademik, kehadiran, dan partisipasi ekstrakurikuler siswa. "
        "Informasi klaster ini dapat digunakan untuk memahami kebutuhan siswa dan "
        "merancang strategi pembinaan yang sesuai."
    )
    pdf.multi_cell(0, 5, keterangan_umum, align='J')
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Nama Siswa: {nama}", ln=True)
    pdf.cell(0, 8, f"Klaster Hasil: {klaster}", ln=True)
    pdf.ln(3)
    klaster_desc = cluster_desc_map.get(klaster, "Deskripsi klaster tidak tersedia.")
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, f"Karakteristik Klaster {klaster}: {klaster_desc}", align='J')
    pdf.ln(5)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    ekskul_diikuti = []
    ekskul_cols_full_names = ["Ekstrakurikuler Komputer", "Ekstrakurikuler Pertanian", "Ekstrakurikuler Menjahit", "Ekstrakurikuler Pramuka"]
    for col in ekskul_cols_full_names:
        val = data_siswa_dict.get(col)
        if val is not None and (val == 1 or str(val).strip() == '1'):
            ekskul_diikuti.append(col.replace("Ekstrakurikuler ", ""))

    display_data = {
        "Nomor Induk": data_siswa_dict.get("No", "-"),
        "Jenis Kelamin": data_siswa_dict.get("JK", "-"),
        "Kelas": data_siswa_dict.get("Kelas", "-"),
        "Rata-rata Nilai Akademik": f"{data_siswa_dict.get('Rata Rata Nilai Akademik', '-'):.2f}",
        "Persentase Kehadiran": f"{data_siswa_dict.get('Kehadiran', '-'):.2%}",
        "Ekstrakurikuler yang Diikuti": ", ".join(ekskul_diikuti) if ekskul_diikuti else "Tidak mengikuti ekstrakurikuler",
    }
    for key, val in display_data.items():
        pdf.cell(0, 7, f"{key}: {val}", ln=True)
    try:
        pdf_output = pdf.output(dest='S').encode('latin-1')
        return bytes(pdf_output)
    except Exception as e:
        st.error(f"Error saat mengonversi PDF: {e}. Coba pastikan tidak ada karakter aneh pada data.")
        return None

def preprocess_data(df):
    df_processed = df.copy()
    df_processed.columns = [col.strip() for col in df_processed.columns]
    missing_cols = [col for col in NUMERIC_COLS + CATEGORICAL_COLS if col not in df_processed.columns]
    if missing_cols:
        st.error(f"Kolom-kolom berikut tidak ditemukan dalam data Anda: {', '.join(missing_cols)}. Harap periksa file Excel Anda dan pastikan nama kolom sudah benar.")
        return None, None
    df_clean_for_clustering = df_processed.drop(columns=ID_COLS, errors="ignore")
    for col in CATEGORICAL_COLS:
        df_clean_for_clustering[col] = df_clean_for_clustering[col].fillna(0).astype(str)
    for col in NUMERIC_COLS:
        if df_clean_for_clustering[col].isnull().any():
            mean_val = df_clean_for_clustering[col].mean()
            df_clean_for_clustering[col] = df_clean_for_clustering[col].fillna(mean_val)
            st.warning(f"Nilai kosong pada kolom '{col}' diisi dengan rata-rata: {mean_val:.2f}.")
    scaler = StandardScaler()
    df_clean_for_clustering[NUMERIC_COLS] = scaler.fit_transform(df_clean_for_clustering[NUMERIC_COLS])
    return df_clean_for_clustering, scaler

def run_kprototypes_clustering(df_preprocessed, n_clusters):
    df_for_clustering = df_preprocessed.copy()
    X_data = df_for_clustering[ALL_FEATURES_FOR_CLUSTERING]
    X = X_data.to_numpy()
    categorical_feature_indices = [X_data.columns.get_loc(c) for c in CATEGORICAL_COLS]
    try:
        kproto = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=10, verbose=0, random_state=42, n_jobs=-1)
        clusters = kproto.fit_predict(X, categorical=categorical_feature_indices)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan K-Prototypes: {e}. Pastikan data Anda cukup bervariasi untuk jumlah klaster yang dipilih.")
        return None, None, None
    df_for_clustering["Klaster"] = clusters
    return df_for_clustering, kproto, categorical_feature_indices

def generate_cluster_descriptions(df_clustered, n_clusters, numeric_cols, categorical_cols):
    cluster_characteristics_map = {}
    if 'df_original' not in st.session_state or st.session_state.df_original is None:
        return {}
    
    df_original_numeric = st.session_state.df_original[NUMERIC_COLS]
    for i in range(n_clusters):
        cluster_data = df_clustered[df_clustered["Klaster"] == i]
        avg_scaled_values = cluster_data[numeric_cols].mean()
        mode_values = cluster_data[categorical_cols].mode().iloc[0]
        desc = ""
        if avg_scaled_values["Rata Rata Nilai Akademik"] > 0.75:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung sangat tinggi. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] > 0.25:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung di atas rata-rata. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] < -0.75:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung sangat rendah. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] < -0.25:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung di bawah rata-rata. "
        else:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung rata-rata. "
        if avg_scaled_values["Kehadiran"] > 0.75:
            desc += "Tingkat kehadiran cenderung sangat tinggi. "
        elif avg_scaled_values["Kehadiran"] > 0.25:
            desc += "Tingkat kehadiran cenderung di atas rata-rata. "
        elif avg_scaled_values["Kehadiran"] < -0.75:
            desc += "Tingkat kehadiran cenderung sangat rendah. "
        elif avg_scaled_values["Kehadiran"] < -0.25:
            desc += "Tingkat kehadiran cenderung di bawah rata-rata. "
        else:
            desc += "Tingkat kehadiran cenderung rata-rata. "
        ekskul_aktif_modes = [col_name for col_name in categorical_cols if mode_values[col_name] == '1']
        if ekskul_aktif_modes:
            desc += f"Siswa di klaster ini aktif dalam ekstrakurikuler: {', '.join([c.replace('Ekstrakurikuler ', '') for c in ekskul_aktif_modes])}."
        else:
            desc += "Siswa di klaster ini kurang aktif dalam kegiatan ekstrakurikuler."
        cluster_characteristics_map[i] = desc
    return cluster_characteristics_map

# --- INISIALISASI SESSION STATE ---
if 'role' not in st.session_state:
    st.session_state.role = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_preprocessed_for_clustering' not in st.session_state:
    st.session_state.df_preprocessed_for_clustering = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'kproto_model' not in st.session_state:
    st.session_state.kproto_model = None
if 'categorical_features_indices' not in st.session_state:
    st.session_state.categorical_features_indices = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3
if 'cluster_characteristics_map' not in st.session_state:
    st.session_state.cluster_characteristics_map = {}
if 'current_menu' not in st.session_state:
    st.session_state.current_menu = None
if 'kepsek_current_menu' not in st.session_state:
    st.session_state.kepsek_current_menu = "Lihat Hasil Klasterisasi"


# --- FUNGSI HALAMAN UTAMA (UNTUK SETIAP PERAN) ---

def show_operator_tu_page():
    st.sidebar.title("MENU NAVIGASI")
    st.sidebar.markdown("---")
    
    menu_options = [
        "Unggah Data",
        "Praproses & Normalisasi Data",
        "Klasterisasi Data K-Prototypes",
        "Prediksi Klaster Siswa Baru",
        "Visualisasi & Profil Klaster",
        "Lihat Profil Siswa Individual"
    ]
    if 'current_menu' not in st.session_state or st.session_state.current_menu not in menu_options:
        st.session_state.current_menu = menu_options[0]

    for option in menu_options:
        icon_map = {
            "Unggah Data": "⬆",
            "Praproses & Normalisasi Data": "⚙",
            "Klasterisasi Data K-Prototypes": "📊",
            "Prediksi Klaster Siswa Baru": "🔮",
            "Visualisasi & Profil Klaster": "📈",
            "Lihat Profil Siswa Individual": "👤"
        }
        display_name = f"{icon_map.get(option, '')} {option}"
        button_key = f"nav_button_{option.replace(' ', '_').replace('&', 'and')}"

        if st.sidebar.button(display_name, key=button_key):
            st.session_state.current_menu = option
            st.rerun()

    # Logika untuk menandai tombol aktif
    js_highlight_active_button = f"""
    <script>
        function cleanButtonText(text) {{
            return (text || '').replace(/\\p{{Emoji}}/gu, '').trim();
        }}
        function highlightActiveSidebarButton() {{
            var currentMenu = '{st.session_state.current_menu}';
            var cleanCurrentMenuName = cleanButtonText(currentMenu);
            var sidebarButtonContainers = window.parent.document.querySelectorAll('[data-testid="stSidebar"] [data-testid="stButton"]');
            sidebarButtonContainers.forEach(function(container) {{
                var button = container.querySelector('button');
                if (button) {{
                    var buttonText = cleanButtonText(button.innerText || button.textContent);
                    container.classList.remove('st-sidebar-button-active');
                    if (buttonText === cleanCurrentMenuName) {{
                        container.classList.add('st-sidebar-button-active');
                    }}
                }}
            }});
        }}
        const observer = new MutationObserver((mutationsList, observer) => {{
            const sidebarChanged = mutationsList.some(mutation =>
                mutation.target.closest('[data-testid="stSidebar"]')
            );
            if (sidebarChanged) {{
                highlightActiveSidebarButton();
            }}
        }});
        observer.observe(window.parent.document.body, {{ childList: true, subtree: true }});
        highlightActiveSidebarButton();
    </script>
    """
    if hasattr(st, 'html'):
        st.html(js_highlight_active_button)
    else:
        st.markdown(js_highlight_active_button, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Keluar", key="logout_tu_sidebar"):
        st.session_state.clear()
        st.rerun()

    if st.session_state.current_menu == "Unggah Data":
        st.header("Unggah Data Siswa")
        st.markdown("""
        <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; border-left: 5px solid #2196F3;'>
        Silakan unggah file Excel (.xlsx) yang berisi dataset siswa. Pastikan file Anda memiliki
        kolom-kolom berikut agar sistem dapat bekerja dengan baik:<br><br>
        <ul>
            <li><b>Kolom Identitas:</b> "No", "Nama", "JK", "Kelas"</li>
            <li><b>Kolom Numerik (untuk analisis):</b> "Rata Rata Nilai Akademik", "Kehadiran"</li>
            <li><b>Kolom Kategorikal (untuk analisis, nilai 0 atau 1):</b> "Ekstrakurikuler Komputer", "Ekstrakurikuler Pertanian", "Ekstrakurikuler Menjahit", "Ekstrakurikuler Pramuka"</li>
        </ul>
        Pastikan nama kolom sudah persis sama dan tidak ada kesalahan penulisan.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        uploaded_file = st.file_uploader("Pilih File Excel Dataset", type=["xlsx"], help="Unggah file Excel Anda di sini. Hanya format .xlsx yang didukung.")
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                st.session_state.df_original = df
                st.session_state.df_clustered = None
                st.success("Data berhasil diunggah! Anda dapat melanjutkan ke langkah praproses.")
                st.subheader("Preview Data yang Diunggah:")
                st.dataframe(df, use_container_width=True, height=300)
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {e}. Pastikan format file Excel benar dan tidak rusak.")

    elif st.session_state.current_menu == "Praproses & Normalisasi Data":
        st.header("Praproses Data & Normalisasi Z-score")
        if st.session_state.df_original is None or st.session_state.df_original.empty:
            st.warning("Silakan unggah data terlebih dahulu di menu 'Unggah Data'.")
        else:
            st.markdown("""
            <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; border-left: 5px solid #2196F3;'>
            Pada tahap ini, data akan disiapkan untuk analisis klasterisasi. Proses yang dilakukan meliputi:
            <ul>
                <li><b>Pembersihan Data:</b> Menangani nilai-nilai yang hilang (missing values) pada kolom numerik (diisi dengan rata-rata).</li>
                <li><b>Konversi Tipe Data:</b> Memastikan kolom kategorikal memiliki tipe data yang sesuai untuk algoritma.</li>
                <li><b>Normalisasi Z-score:</b> Mengubah skala fitur numerik (nilai akademik & kehadiran) agar memiliki rata-rata nol dan deviasi standar satu, sehingga semua fitur memiliki bobot yang setara dalam perhitungan klasterisasi.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            if st.button("Jalankan Praproses & Normalisasi"):
                with st.spinner("Sedang memproses dan menormalisasi data..."):
                    df_preprocessed, scaler = preprocess_data(st.session_state.df_original)
                if df_preprocessed is not None and scaler is not None:
                    st.session_state.df_preprocessed_for_clustering = df_preprocessed
                    st.session_state.scaler = scaler
                    st.success("Praproses dan Normalisasi berhasil dilakukan. Data siap untuk klasterisasi!")
                    st.subheader("Data Setelah Praproses dan Normalisasi:")
                    st.dataframe(st.session_state.df_preprocessed_for_clustering, use_container_width=True, height=300)
                    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    elif st.session_state.current_menu == "Klasterisasi Data K-Prototypes":
        st.header("Klasterisasi K-Prototypes")
        if st.session_state.df_preprocessed_for_clustering is None or st.session_state.df_preprocessed_for_clustering.empty:
            st.warning("Silakan lakukan praproses data terlebih dahulu di menu 'Praproses & Normalisasi Data'.")
        else:
            st.markdown("""
            <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; border-left: 5px solid #2196F3;'>
            Pada tahap ini, Anda akan menjalankan algoritma K-Prototypes untuk mengelompokkan siswa.
            <br><br>
            Pilih <b>Jumlah Klaster (K)</b> yang Anda inginkan (antara 2 hingga 6). Algoritma ini akan
            mengelompokkan siswa berdasarkan kombinasi fitur numerik (nilai akademik, kehadiran) dan
            fitur kategorikal (ekstrakurikuler) yang telah disiapkan sebelumnya.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            k = st.slider("Pilih Jumlah Klaster (K)", 2, 6, value=st.session_state.n_clusters,
                            help="Pilih berapa banyak kelompok siswa yang ingin Anda bentuk.")
            if st.button("Jalankan Klasterisasi"):
                with st.spinner(f"Melakukan klasterisasi dengan {k} klaster..."):
                    df_clustered, kproto_model, categorical_features_indices = run_kprototypes_clustering(
                        st.session_state.df_preprocessed_for_clustering, k
                    )
                if df_clustered is not None:
                    df_final = st.session_state.df_original.copy()
                    df_final['Klaster'] = df_clustered['Klaster']
                    st.session_state.df_clustered = df_final
                    st.session_state.kproto_model = kproto_model
                    st.session_state.categorical_features_indices = categorical_features_indices
                    st.session_state.n_clusters = k
                    st.session_state.cluster_characteristics_map = generate_cluster_descriptions(
                        df_clustered, k, NUMERIC_COLS, CATEGORICAL_COLS
                    )
                    st.success(f"Klasterisasi selesai dengan {k} klaster! Hasil pengelompokan siswa telah tersedia.")
                    st.markdown("---")
                    st.subheader("Data Hasil Klasterisasi (Disertai Data Asli):")
                    st.dataframe(df_final, use_container_width=True, height=300)
                    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
                    st.subheader("Ringkasan Klaster: Jumlah Siswa per Kelompok")
                    jumlah_per_klaster = df_final["Klaster"].value_counts().sort_index().reset_index()
                    jumlah_per_klaster.columns = ["Klaster", "Jumlah Siswa"]
                    st.table(jumlah_per_klaster)
                    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
                    st.subheader(f"Karakteristik Umum Klaster ({st.session_state.n_clusters} Klaster):")
                    st.write("Berikut adalah deskripsi singkat untuk setiap klaster yang terbentuk:")
                    for cluster_id, desc in st.session_state.cluster_characteristics_map.items():
                        with st.expander(f"Klaster {cluster_id}"):
                            st.markdown(desc)
                    
                    try:
                        df_final_for_kepsek = df_final.copy()
                        df_final_for_kepsek['Kehadiran'] = df_final_for_kepsek['Kehadiran'].apply(lambda x: f"{x:.2%}")
                        file_name = "Data MA-ALHIKMAH.xlsx"
                        df_final_for_kepsek.to_excel(file_name, index=False)
                        st.success(f"Hasil klasterisasi berhasil disimpan ke file '{file_name}' untuk diakses oleh Kepala Sekolah.")
                    except Exception as e:
                        st.error(f"Gagal menyimpan file Excel untuk Kepala Sekolah: {e}")

    elif st.session_state.current_menu == "Prediksi Klaster Siswa Baru":
        st.header("Prediksi Klaster untuk Siswa Baru")
        if st.session_state.kproto_model is None or st.session_state.scaler is None:
            st.warning("Silakan lakukan klasterisasi terlebih dahulu di menu 'Klasterisasi Data K-Prototypes' untuk melatih model dan scaler.")
        else:
            st.markdown("""
            <div style='background-color:#f1f9ff; padding:15px; border-radius:10px; border-left: 5px solid #2C2F7F;'>
            Halaman ini memungkinkan Anda untuk memprediksi klaster bagi siswa baru. Masukkan data nilai akademik,
            kehadiran, dan keterlibatan ekstrakurikuler siswa. Sistem akan otomatis memproses data
            dan memetakan siswa ke klaster yang paling sesuai berdasarkan model yang telah dilatih.
            <br><br>
            Pemanfaatan klaster membantu guru dalam merancang strategi pembinaan dan pendekatan pembelajaran
            yang lebih personal dan efektif.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            with st.form("form_input_siswa_baru", clear_on_submit=False):
                st.markdown("### Input Data Siswa Baru")
                st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Data Akademik & Kehadiran")
                    input_rata_nilai = st.number_input("Rata-rata Nilai Akademik (0 - 100)", min_value=0.0, max_value=100.0, value=None, placeholder="Contoh: 85.5", format="%.2f", key="input_nilai_prediksi")
                    input_kehadiran = st.number_input("Persentase Kehadiran (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=None, placeholder="Contoh: 0.95 (untuk 95%)", format="%.2f", key="input_kehadiran_prediksi")
                with col2:
                    st.markdown("#### Keikutsertaan Ekstrakurikuler")
                    st.write("Centang ekstrakurikuler yang diikuti siswa:")
                    input_cat_ekskul_values = []
                    for idx, col in enumerate(CATEGORICAL_COLS):
                        val = st.checkbox(col.replace("Ekstrakurikuler ", ""), key=f"ekskul_prediksi_{idx}")
                        input_cat_ekskul_values.append(1 if val else 0)
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Prediksi Klaster Siswa")
            if submitted:
                if input_rata_nilai is None or input_kehadiran is None:
                    st.error("Harap isi semua nilai numerik (Rata-rata Nilai Akademik dan Persentase Kehadiran) terlebih dahulu.")
                else:
                    input_numeric_data = [input_rata_nilai, input_kehadiran]
                    normalized_numeric_data = st.session_state.scaler.transform([input_numeric_data])[0]
                    new_student_data_for_prediction = np.array(
                        list(normalized_numeric_data) + input_cat_ekskul_values, dtype=object
                    ).reshape(1, -1)
                    predicted_cluster = st.session_state.kproto_model.predict(
                        new_student_data_for_prediction, categorical=st.session_state.categorical_features_indices
                    )
                    st.success(f"Prediksi Klaster: Siswa Baru Ini Masuk ke Klaster {predicted_cluster[0]}!")
                    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                    klaster_desc_for_new_student = st.session_state.cluster_characteristics_map.get(predicted_cluster[0], "Deskripsi klaster tidak tersedia.")
                    st.markdown(f"""
                    <div style='background-color:#e8f5e9; padding:15px; border-radius:10px; border-left: 5px solid #4CAF50;'>
                    <b>Karakteristik Klaster {predicted_cluster[0]}:</b><br>
                    {klaster_desc_for_new_student}
                    <br><br>
                    Informasi ini sangat membantu guru dalam memberikan bimbingan dan dukungan yang tepat sasaran
                    sesuai dengan profil klaster siswa.
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
                    st.subheader("Visualisasi Karakteristik Siswa Baru (Dinormalisasi)")
                    st.write("Grafik ini menampilkan nilai fitur siswa setelah dinormalisasi (nilai akademik & kehadiran) atau dalam format biner (ekstrakurikuler).")
                    values_for_plot = list(normalized_numeric_data) + input_cat_ekskul_values
                    labels_for_plot = ["Nilai Akademik (Norm)", "Kehadiran (Norm)"] + [col.replace("Ekstrakurikuler ", "Ekskul\n") for col in CATEGORICAL_COLS]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = sns.barplot(x=labels_for_plot, y=values_for_plot, palette="viridis", ax=ax)
                    ax.set_ylim(min(values_for_plot) - 0.2 if values_for_plot else -1, max(values_for_plot) + 0.2 if values_for_plot else 1)
                    for index, value in enumerate(values_for_plot):
                        ax.text(bars.patches[index].get_x() + bars.patches[index].get_width() / 2,
                                bars.patches[index].get_height() + (0.05 if value >= 0 else -0.1),
                                f"{value:.2f}", ha='center', fontsize=9, weight='bold')
                    ax.set_title("Profil Siswa Baru", fontsize=16, weight='bold')
                    ax.set_ylabel("Nilai (Dinormalisasi / Biner)")
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)

    elif st.session_state.current_menu == "Visualisasi & Profil Klaster":
        st.header("Visualisasi dan Interpretasi Profil Klaster")
        if st.session_state.df_preprocessed_for_clustering is None or st.session_state.df_preprocessed_for_clustering.empty:
            st.warning("Silakan unggah data dan lakukan praproses terlebih dahulu di menu 'Praproses & Normalisasi Data'.")
        else:
            st.markdown("""
            <div style='background-color:#f1f9ff; padding:15px; border-radius:10px; border-left: 5px solid #2C2F7F;'>
            Di halaman ini, Anda dapat memilih jumlah klaster (K) dan melihat visualisasi serta ringkasan
            karakteristik dari setiap kelompok siswa. Visualisasi ini dirancang untuk membantu Anda
            memahami perbedaan utama antara klaster-klaster yang terbentuk.
            <br><br>
            Setiap bar pada grafik merepresentasikan rata-rata (untuk fitur numerik yang dinormalisasi)
            atau modus (untuk fitur kategorikal biner 0/1) dari fitur-fitur di dalam klaster tersebut.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            k_visual = st.slider("Jumlah Klaster (K) untuk visualisasi", 2, 6, value=st.session_state.n_clusters,
                                 help="Geser untuk memilih jumlah klaster yang ingin Anda visualisasikan. Ini akan melatih ulang model sementara untuk tujuan visualisasi.")
            df_for_visual_clustering, kproto_visual, cat_indices_visual = run_kprototypes_clustering(
                st.session_state.df_preprocessed_for_clustering, k_visual
            )
            if df_for_visual_clustering is not None:
                cluster_characteristics_map_visual = generate_cluster_descriptions(
                    df_for_visual_clustering, k_visual, NUMERIC_COLS, CATEGORICAL_COLS
                )
                st.markdown(f"### Menampilkan Profil Klaster untuk K = {k_visual}")
                st.write("Visualisasi ini menggunakan data yang telah dinormalisasi (nilai, kehadiran) atau dikodekan (ekstrakurikuler 0/1).")
                for i in range(k_visual):
                    st.markdown(f"---")
                    st.subheader(f"Klaster {i}")
                    cluster_data = df_for_visual_clustering[df_for_visual_clustering["Klaster"] == i]
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("#### Statistik Klaster")
                        st.markdown(f"Jumlah Siswa: {len(cluster_data)}")
                        st.write("Rata-rata Nilai & Kehadiran (Dinormalisasi):")
                        st.dataframe(cluster_data[NUMERIC_COLS].mean().round(2).to_frame(name='Rata-rata'), use_container_width=True)
                        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                        st.write("Kecenderungan Ekstrakurikuler (Modus):")
                        mode_ekskul_display = cluster_data[CATEGORICAL_COLS].mode().iloc[0].apply(lambda x: 'Ya' if x == '1' else 'Tidak')
                        st.dataframe(mode_ekskul_display.to_frame(name='Paling Umum'), use_container_width=True)
                        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                        st.info(f"Ringkasan Karakteristik Klaster {i}:\n{cluster_characteristics_map_visual.get(i, 'Deskripsi tidak tersedia.')}")
                    with col2:
                        st.markdown("#### Grafik Profil Klaster")
                        st.write("📈 Visualisasi ini menunjukkan rata-rata (numerik) atau modus (kategorikal) dari fitur-fitur di klaster ini.")
                        values_for_plot_numeric = cluster_data[NUMERIC_COLS].mean().tolist()
                        values_for_plot_ekskul = [int(cluster_data[col].mode().iloc[0]) for col in CATEGORICAL_COLS]
                        values_for_plot = values_for_plot_numeric + values_for_plot_ekskul
                        labels_for_plot = ["Nilai (Norm)", "Kehadiran (Norm)"] + [col.replace("Ekstrakurikuler ", "Ekskul\n") for col in CATEGORICAL_COLS]
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = sns.barplot(x=labels_for_plot, y=values_for_plot, palette="cubehelix", ax=ax)
                        ax.set_ylim(min(values_for_plot) - 0.2 if values_for_plot else -1, max(values_for_plot) + 0.2 if values_for_plot else 1)
                        for index, value in enumerate(values_for_plot):
                            offset = 0.05 if value >= 0 else -0.1
                            ax.text(bars.patches[index].get_x() + bars.patches[index].get_width() / 2, bars.patches[index].get_height() + offset, f"{value:.2f}", ha='center', fontsize=9, weight='bold')
                        ax.set_title(f"Profil Klaster {i}", fontsize=16, weight='bold')
                        ax.set_ylabel("Nilai (Dinormalisasi / Biner)")
                        plt.xticks(rotation=0)
                        plt.tight_layout()
                        st.pyplot(fig)
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    elif st.session_state.current_menu == "Lihat Profil Siswa Individual":
        st.header("Lihat Profil Siswa Berdasarkan Nama")
        if st.session_state.df_clustered is None or st.session_state.df_original is None or st.session_state.df_original.empty:
            st.warning("Silakan unggah data di menu 'Unggah Data' dan lakukan klasterisasi di menu 'Klasterisasi Data K-Prototypes' terlebih dahulu.")
        else:
            st.info("Pilih nama siswa dari daftar di bawah untuk melihat detail profil mereka, termasuk klaster tempat mereka berada dan karakteristiknya.")
            st.markdown("---")
            df_original_with_cluster = st.session_state.df_clustered
            default_index = 0
            if "selected_student_name" in st.session_state and st.session_state.selected_student_name in df_original_with_cluster["Nama"].unique():
                try:
                    default_index = list(df_original_with_cluster["Nama"].unique()).index(st.session_state.selected_student_name)
                except ValueError:
                    default_index = 0
            nama_terpilih = st.selectbox(
                "Pilih Nama Siswa",
                df_original_with_cluster["Nama"].unique(),
                index=default_index,
                key="pilih_nama_siswa_selectbox_tu",
                help="Pilih siswa yang profilnya ingin Anda lihat."
            )
            st.session_state.selected_student_name = nama_terpilih
            if nama_terpilih:
                siswa_data = df_original_with_cluster[df_original_with_cluster["Nama"] == nama_terpilih].iloc[0]
                klaster_siswa_terpilih = siswa_data['Klaster']
                st.success(f"Siswa {nama_terpilih} tergolong dalam Klaster {klaster_siswa_terpilih} (hasil dari {st.session_state.n_clusters} klaster).")
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                klaster_desc_for_new_student = st.session_state.cluster_characteristics_map.get(klaster_siswa_terpilih, "Deskripsi klaster tidak tersedia.")
                st.markdown(f"""
                <div style='background-color:#f0f4f7; padding:15px; border-radius:10px; border-left: 5px solid {PRIMARY_COLOR};'>
                <b>Karakteristik Klaster Ini:</b><br>
                {klaster_desc_for_new_student}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("---")
                st.subheader("Detail Data Siswa")
                col_info, col_chart = st.columns([1, 2])
                with col_info:
                    st.markdown("#### Informasi Dasar")
                    st.markdown(f"Nomor Induk: {siswa_data.get('No', '-')}")
                    st.markdown(f"Jenis Kelamin: {siswa_data.get('JK', '-')}")
                    st.markdown(f"Kelas: {siswa_data.get('Kelas', '-')}")
                    st.markdown(f"Rata-rata Nilai Akademik: {siswa_data.get('Rata Rata Nilai Akademik', '-'):.2f}")
                    st.markdown(f"Persentase Kehadiran: {siswa_data.get('Kehadiran', '-'):.2%}")
                    st.markdown("#### Ekstrakurikuler yang Diikuti")
                    ekskul_diikuti_str = []
                    for col in CATEGORICAL_COLS:
                        if siswa_data.get(col, 0) == 1:
                            ekskul_diikuti_str.append(col.replace("Ekstrakurikuler ", ""))
                    if ekskul_diikuti_str:
                        for ekskul in ekskul_diikuti_str:
                            st.markdown(f"- {ekskul} ✅")
                    else:
                        st.markdown("Tidak mengikuti ekstrakurikuler ❌")
                with col_chart:
                    st.markdown("#### Visualisasi Profil Siswa Individual")
                    st.write("Grafik ini menampilkan nilai asli (tidak dinormalisasi) untuk rata-rata nilai akademik dan persentase kehadiran (0-100%), serta status biner (0/1) untuk ekstrakurikuler.")
                    labels_siswa_plot = ["Rata-rata\nNilai Akademik", "Kehadiran (%)"] + [col.replace("Ekstrakurikuler ", "Ekskul\n") for col in CATEGORICAL_COLS]
                    values_siswa_plot_numeric = [
                        siswa_data["Rata Rata Nilai Akademik"],
                        siswa_data["Kehadiran"] * 100
                    ]
                    values_siswa_plot_ekskul = [
                        siswa_data[col] * 100 for col in CATEGORICAL_COLS
                    ]
                    values_siswa_plot = values_siswa_plot_numeric + values_siswa_plot_ekskul
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = sns.barplot(x=labels_siswa_plot, y=values_siswa_plot, palette="magma", ax=ax)
                    max_plot_val = max(values_siswa_plot) if values_siswa_plot else 100
                    ax.set_ylim(0, max(100, max_plot_val * 1.1))
                    for bar, val in zip(bars.patches, values_siswa_plot):
                        ax.text(bar.get_x() + bar.get_width() / 2, val + (ax.get_ylim()[1] * 0.02), f"{val:.1f}", ha='center', fontsize=9, weight='bold')
                    ax.set_title(f"Grafik Profil Siswa - {nama_terpilih}", fontsize=16, weight='bold')
                    ax.set_ylabel("Nilai / Status (%)")
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
                st.subheader(f"Siswa Lain di Klaster {klaster_siswa_terpilih}:")
                siswa_lain_di_klaster = df_original_with_cluster[
                    (df_original_with_cluster['Klaster'] == klaster_siswa_terpilih) &
                    (df_original_with_cluster['Nama'] != nama_terpilih)
                ]
                if not siswa_lain_di_klaster.empty:
                    st.write("Berikut adalah daftar siswa lain yang juga tergolong dalam klaster ini:")
                    display_cols_for_others = ["No", "Nama", "JK", "Kelas", "Rata Rata Nilai Akademik", "Kehadiran"]
                    display_df_others = siswa_lain_di_klaster[display_cols_for_others].copy()
                    display_df_others["Kehadiran"] = display_df_others["Kehadiran"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(display_df_others, use_container_width=True)
                else:
                    st.info("Tidak ada siswa lain dalam klaster ini.")
                st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
                st.subheader("Unduh Laporan Profil Siswa (PDF)")
                if st.session_state.cluster_characteristics_map:
                    if st.button("Generate & Unduh Laporan PDF", key="unduh_pdf_tu", help="Klik untuk membuat laporan PDF profil siswa ini."):
                        with st.spinner("Menyiapkan laporan PDF..."):
                            siswa_data_for_pdf = siswa_data.drop(labels=["Klaster"]).to_dict()
                            pdf_data_bytes = generate_pdf_profil_siswa(
                                nama_terpilih,
                                siswa_data_for_pdf,
                                siswa_data["Klaster"],
                                st.session_state.cluster_characteristics_map
                            )
                        if pdf_data_bytes:
                            st.success("Laporan PDF berhasil disiapkan!")
                            st.download_button(
                                label="Klik di Sini untuk Mengunduh PDF",
                                data=pdf_data_bytes,
                                file_name=f"Profil_{nama_terpilih.replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                key="download_profile_pdf_tu_final",
                                help="Klik ini untuk menyimpan laporan PDF ke perangkat Anda."
                            )
                else:
                    st.warning("Mohon lakukan klasterisasi terlebih dahulu (Menu 'Klasterisasi Data K-Prototypes') untuk menghasilkan data profil PDF.")


def show_kepala_sekolah_page():
    file_path = "Data MA-ALHIKMAH.xlsx"
    df_kepsek_load = None
    if os.path.exists(file_path):
        try:
            df_kepsek_load = pd.read_excel(file_path, engine='openpyxl')
            st.session_state.df_clustered = df_kepsek_load
            
            if 'df_original' not in st.session_state or st.session_state.df_original is None:
                df_original_from_clustered = df_kepsek_load.copy()
                if df_original_from_clustered['Kehadiran'].dtype == 'object':
                    df_original_from_clustered['Kehadiran'] = df_original_from_clustered['Kehadiran'].str.rstrip('%').astype('float') / 100
                st.session_state.df_original = df_original_from_clustered.drop(columns=['Klaster'], errors='ignore')

                n_clusters_kepsek = len(df_kepsek_load['Klaster'].unique())
                st.session_state.n_clusters = n_clusters_kepsek
                
                df_preprocessed, scaler = preprocess_data(st.session_state.df_original)
                if df_preprocessed is not None:
                    df_preprocessed['Klaster'] = df_kepsek_load['Klaster']
                    st.session_state.cluster_characteristics_map = generate_cluster_descriptions(
                        df_preprocessed, n_clusters_kepsek, NUMERIC_COLS, CATEGORICAL_COLS
                    )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file '{file_path}': {e}.")
            st.session_state.df_clustered = None
            
    st.sidebar.title("MENU NAVIGASI")
    st.sidebar.markdown("---")
    
    kepsek_menu_options = [
        "Lihat Hasil Klasterisasi",
        "Visualisasi & Profil Klaster",
        "Lihat Profil Siswa Individual"
    ]
    if 'kepsek_current_menu' not in st.session_state:
        st.session_state.kepsek_current_menu = kepsek_menu_options[0]

    for option in kepsek_menu_options:
        icon_map = {
            "Lihat Hasil Klasterisasi": "📋",
            "Visualisasi & Profil Klaster": "📈",
            "Lihat Profil Siswa Individual": "👤"
        }
        display_name = f"{icon_map.get(option, '')} {option}"
        button_key = f"kepsek_nav_button_{option.replace(' ', '_').replace('&', 'and')}"

        if st.sidebar.button(display_name, key=button_key):
            st.session_state.kepsek_current_menu = option
            st.rerun()

    js_highlight_active_button = f"""
    <script>
        function cleanButtonText(text) {{
            return (text || '').replace(/\\p{{Emoji}}/gu, '').trim();
        }}
        function highlightActiveSidebarButton() {{
            var currentMenu = '{st.session_state.kepsek_current_menu}';
            var cleanCurrentMenuName = cleanButtonText(currentMenu);
            var sidebarButtonContainers = window.parent.document.querySelectorAll('[data-testid="stSidebar"] [data-testid="stButton"]');
            sidebarButtonContainers.forEach(function(container) {{
                var button = container.querySelector('button');
                if (button) {{
                    var buttonText = cleanButtonText(button.innerText || button.textContent);
                    container.classList.remove('st-sidebar-button-active');
                    if (buttonText === cleanCurrentMenuName) {{
                        container.classList.add('st-sidebar-button-active');
                    }}
                }}
            }});
        }}
        const observer = new MutationObserver((mutationsList, observer) => {{
            const sidebarChanged = mutationsList.some(mutation =>
                mutation.target.closest('[data-testid="stSidebar"]')
            );
            if (sidebarChanged) {{
                highlightActiveSidebarButton();
            }}
        }});
        observer.observe(window.parent.document.body, {{ childList: true, subtree: true }});
        highlightActiveSidebarButton();
    </script>
    """
    if hasattr(st, 'html'):
        st.html(js_highlight_active_button)
    else:
        st.markdown(js_highlight_active_button, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Keluar", key="logout_kepsek_sidebar"):
        st.session_state.clear()
        st.rerun()
    
    st.title("👨‍💼 Dasbor Kepala Sekolah")
    
    if st.session_state.df_clustered is None or st.session_state.df_clustered.empty:
        st.warning(f"File hasil klasterisasi 'Data MA-ALHIKMAH.xlsx' tidak ditemukan atau tidak valid. Mohon minta Operator TU untuk memproses dan menyimpan hasilnya terlebih dahulu.")
        return

    if st.session_state.kepsek_current_menu == "Lihat Hasil Klasterisasi":
        st.header("Hasil Klasterisasi Siswa")
        st.info("Halaman ini menampilkan data siswa yang sudah dikelompokkan ke dalam klaster.")
        st.markdown("---")
        
        st.subheader("Data Hasil Klasterisasi")
        st.dataframe(st.session_state.df_clustered, use_container_width=True, height=300)
        
        st.markdown("---")
        st.subheader("Ringkasan Klaster: Jumlah Siswa per Kelompok")
        jumlah_per_klaster = st.session_state.df_clustered["Klaster"].value_counts().sort_index().reset_index()
        jumlah_per_klaster.columns = ["Klaster", "Jumlah Siswa"]
        st.table(jumlah_per_klaster)
    
    elif st.session_state.kepsek_current_menu == "Visualisasi & Profil Klaster":
        st.header("Visualisasi dan Interpretasi Profil Klaster")
        st.info("Anda dapat melihat visualisasi dan ringkasan karakteristik dari setiap kelompok siswa.")
        st.markdown("---")
        
        if not st.session_state.cluster_characteristics_map:
            st.warning("Deskripsi klaster tidak tersedia. Mohon Operator TU memproses data terlebih dahulu.")
            return

        st.subheader(f"Karakteristik Umum Klaster ({st.session_state.n_clusters} Klaster):")
        st.write("Berikut adalah deskripsi singkat untuk setiap klaster yang terbentuk:")
        
        for i in range(st.session_state.n_clusters):
            st.markdown(f"---")
            st.subheader(f"Klaster {i}")
            cluster_data = st.session_state.df_clustered[st.session_state.df_clustered["Klaster"] == i]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Statistik Klaster")
                st.markdown(f"Jumlah Siswa: {len(cluster_data)}")
                
                df_preprocessed_temp, scaler_temp = preprocess_data(st.session_state.df_original)
                if df_preprocessed_temp is not None:
                    df_preprocessed_temp['Klaster'] = st.session_state.df_clustered['Klaster']
                    cluster_data_norm = df_preprocessed_temp[df_preprocessed_temp["Klaster"] == i]
                    st.write("Rata-rata Nilai & Kehadiran (Dinormalisasi):")
                    st.dataframe(cluster_data_norm[NUMERIC_COLS].mean().round(2).to_frame(name='Rata-rata'), use_container_width=True)

                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                st.write("Kecenderungan Ekstrakurikuler (Modus):")
                mode_ekskul_display = cluster_data[CATEGORICAL_COLS].mode().iloc[0].apply(lambda x: 'Ya' if x == '1' else 'Tidak')
                st.dataframe(mode_ekskul_display.to_frame(name='Paling Umum'), use_container_width=True)
                
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                st.info(f"Ringkasan Karakteristik Klaster {i}:\n{st.session_state.cluster_characteristics_map.get(i, 'Deskripsi tidak tersedia.')}")
            
            with col2:
                st.markdown("#### Grafik Profil Klaster")
                st.write("📈 Visualisasi ini menunjukkan rata-rata (numerik) atau modus (kategorikal) dari fitur-fitur di klaster ini.")
                
                if df_preprocessed_temp is not None:
                    values_for_plot_numeric = cluster_data_norm[NUMERIC_COLS].mean().tolist()
                    values_for_plot_ekskul = [int(cluster_data_norm[col].mode().iloc[0]) for col in CATEGORICAL_COLS]
                    values_for_plot = values_for_plot_numeric + values_for_plot_ekskul
                    labels_for_plot = ["Nilai (Norm)", "Kehadiran (Norm)"] + [col.replace("Ekstrakurikuler ", "Ekskul\n") for col in CATEGORICAL_COLS]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = sns.barplot(x=labels_for_plot, y=values_for_plot, palette="cubehelix", ax=ax)
                    ax.set_ylim(min(values_for_plot) - 0.2 if values_for_plot else -1, max(values_for_plot) + 0.2 if values_for_plot else 1)
                    
                    for index, value in enumerate(values_for_plot):
                        offset = 0.05 if value >= 0 else -0.1
                        ax.text(bars.patches[index].get_x() + bars.patches[index].get_width() / 2, bars.patches[index].get_height() + offset, f"{value:.2f}", ha='center', fontsize=9, weight='bold')
                    
                    ax.set_title(f"Profil Klaster {i}", fontsize=16, weight='bold')
                    ax.set_ylabel("Nilai (Dinormalisasi / Biner)")
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
    elif st.session_state.kepsek_current_menu == "Lihat Profil Siswa Individual":
        st.header("Lihat Profil Siswa Berdasarkan Nama")
        st.info("Pilih nama siswa dari daftar di bawah untuk melihat detail profil mereka, termasuk klaster tempat mereka berada dan karakteristiknya.")
        st.markdown("---")

        df_kepsek = st.session_state.df_clustered
        default_index = 0
        if "selected_student_name_kepsek" in st.session_state and st.session_state.selected_student_name_kepsek in df_kepsek["Nama"].unique():
            try:
                default_index = list(df_kepsek["Nama"].unique()).index(st.session_state.selected_student_name_kepsek)
            except ValueError:
                default_index = 0
        nama_terpilih_kepsek = st.selectbox(
            "Pilih Nama Siswa",
            df_kepsek["Nama"].unique(),
            index=default_index,
            key="pilih_nama_siswa_kepsek",
            help="Pilih siswa yang profilnya ingin Anda lihat."
        )
        st.session_state.selected_student_name_kepsek = nama_terpilih_kepsek
        
        if nama_terpilih_kepsek:
            siswa_data = df_kepsek[df_kepsek["Nama"] == nama_terpilih_kepsek].iloc[0]
            klaster_siswa_terpilih = siswa_data['Klaster']
            st.success(f"Siswa {nama_terpilih_kepsek} tergolong dalam Klaster {klaster_siswa_terpilih}.")
            klaster_desc_for_new_student = st.session_state.cluster_characteristics_map.get(klaster_siswa_terpilih, "Deskripsi klaster tidak tersedia.")
            st.markdown(f"""
            <div style='background-color:#f0f4f7; padding:15px; border-radius:10px; border-left: 5px solid {PRIMARY_COLOR};'>
            <b>Karakteristik Klaster Ini:</b><br>
            {klaster_desc_for_new_student}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("Detail Data Siswa")
            col_info, col_chart = st.columns([1, 2])
            with col_info:
                st.markdown("#### Informasi Dasar")
                st.markdown(f"Nomor Induk: {siswa_data.get('No', '-')}")
                st.markdown(f"Jenis Kelamin: {siswa_data.get('JK', '-')}")
                st.markdown(f"Kelas: {siswa_data.get('Kelas', '-')}")
                st.markdown(f"Rata-rata Nilai Akademik: {siswa_data.get('Rata Rata Nilai Akademik', '-'):.2f}")
                st.markdown(f"Persentase Kehadiran: {siswa_data.get('Kehadiran', '-')}")
                st.markdown("#### Ekstrakurikuler yang Diikuti")
                ekskul_diikuti_str = []
                for col in CATEGORICAL_COLS:
                    if siswa_data.get(col, 0) == 1 or siswa_data.get(col, '0') == '1':
                        ekskul_diikuti_str.append(col.replace("Ekstrakurikuler ", ""))
                if ekskul_diikuti_str:
                    for ekskul in ekskul_diikuti_str:
                        st.markdown(f"- {ekskul} ✅")
                else:
                    st.markdown("Tidak mengikuti ekstrakurikuler ❌")
            with col_chart:
                st.markdown("#### Visualisasi Profil Siswa Individual")
                st.write("Grafik ini menampilkan nilai asli (tidak dinormalisasi) untuk rata-rata nilai akademik dan persentase kehadiran (0-100%), serta status biner (0/1) untuk ekstrakurikuler.")
                labels_siswa_plot = ["Rata-rata\nNilai Akademik", "Kehadiran (%)"] + [col.replace("Ekstrakurikuler ", "Ekskul\n") for col in CATEGORICAL_COLS]
                values_siswa_plot_numeric = [
                    siswa_data.get("Rata Rata Nilai Akademik", 0),
                    float(str(siswa_data.get("Kehadiran", "0%")).replace('%',''))
                ]
                values_siswa_plot_ekskul = [
                    siswa_data.get(col, 0) * 100 for col in CATEGORICAL_COLS
                ]
                values_siswa_plot = values_siswa_plot_numeric + values_siswa_plot_ekskul
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(x=labels_siswa_plot, y=values_siswa_plot, palette="magma", ax=ax)
                max_plot_val = max(values_siswa_plot) if values_siswa_plot else 100
                ax.set_ylim(0, max(100, max_plot_val * 1.1))
                for bar, val in zip(bars.patches, values_siswa_plot):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + (ax.get_ylim()[1] * 0.02), f"{val:.1f}", ha='center', fontsize=9, weight='bold')
                ax.set_title(f"Grafik Profil Siswa - {nama_terpilih_kepsek}", fontsize=16, weight='bold')
                ax.set_ylabel("Nilai / Status (%)")
                plt.xticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
            st.markdown("---")
            st.subheader(f"Siswa Lain di Klaster {klaster_siswa_terpilih}:")
            siswa_lain_di_klaster = df_kepsek[
                (df_kepsek['Klaster'] == klaster_siswa_terpilih) &
                (df_kepsek['Nama'] != nama_terpilih_kepsek)
            ]
            if not siswa_lain_di_klaster.empty:
                st.write("Berikut adalah daftar siswa lain yang juga tergolong dalam klaster ini:")
                display_cols_for_others = ["No", "Nama", "JK", "Kelas", "Rata Rata Nilai Akademik", "Kehadiran"]
                st.dataframe(siswa_lain_di_klaster[display_cols_for_others], use_container_width=True)
            else:
                st.info("Tidak ada siswa lain dalam klaster ini.")
            st.markdown("---")
            st.subheader("Unduh Laporan Profil Siswa (PDF)")
            if st.session_state.cluster_characteristics_map:
                if st.button("Generate & Unduh Laporan PDF", key="unduh_pdf_kepsek", help="Klik untuk membuat laporan PDF profil siswa ini."):
                    with st.spinner("Menyiapkan laporan PDF..."):
                        siswa_data_for_pdf = siswa_data.drop(labels=["Klaster"]).to_dict()
                        if isinstance(siswa_data_for_pdf.get('Kehadiran'), str):
                            siswa_data_for_pdf['Kehadiran'] = float(siswa_data_for_pdf['Kehadiran'].replace('%', '')) / 100
                        pdf_data_bytes = generate_pdf_profil_siswa(
                            nama_terpilih_kepsek,
                            siswa_data_for_pdf,
                            siswa_data["Klaster"],
                            st.session_state.cluster_characteristics_map
                        )
                    if pdf_data_bytes:
                        st.success("Laporan PDF berhasil disiapkan!")
                        st.download_button(
                            label="Klik di Sini untuk Mengunduh PDF",
                            data=pdf_data_bytes,
                            file_name=f"Profil_{nama_terpilih_kepsek.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            key="download_profile_pdf_kepsek_final",
                            help="Klik ini untuk menyimpan laporan PDF ke perangkat Anda."
                        )
            else:
                st.warning("Data klasterisasi tidak valid untuk membuat profil PDF.")


# --- LOGIKA UTAMA APLIKASI UNTUK PEMILIHAN PERAN ---

if st.session_state.role is None:
    st.sidebar.empty()
    st.markdown("""
    <div class="login-container">
        <div class="login-card">
            <h2>Pilih Peran Anda</h2>
            <p style='margin-bottom: 25px;'>Selamat datang di sistem pengelompokan siswa. Silakan pilih peran Anda untuk melanjutkan.</p>
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1;">
                    <style>
                        .st-emotion-cache-199v5-container > button {{
                            background-color: {PRIMARY_COLOR};
                            color: white;
                            width: 100%;
                            font-size: 1.2em;
                            padding: 15px 0;
                            font-weight: bold;
                        }}
                    </style>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_tu, col_kepsek = st.columns(2)
    with col_tu:
        if st.button("Masuk sebagai **Operator TU**", use_container_width=True, key="login_tu"):
            st.session_state.role = 'Operator TU'
            st.session_state.current_menu = "Unggah Data"
            st.rerun()
    with col_kepsek:
        if st.button("Masuk sebagai **Kepala Sekolah**", use_container_width=True, key="login_kepsek"):
            st.session_state.role = 'Kepala Sekolah'
            st.session_state.kepsek_current_menu = "Lihat Hasil Klasterisasi"
            st.rerun()
            
elif st.session_state.role == 'Operator TU':
    show_operator_tu_page()

elif st.session_state.role == 'Kepala Sekolah':
    show_kepala_sekolah_page()