import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io

# Set matplotlib backend to avoid GUI issues in deployment
import matplotlib
matplotlib.use('Agg')

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
    [data-testid="stSidebar"] {{
        background-color: {PRIMARY_COLOR};
        color: #ffffff;
        padding-top: 2.5rem;
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff;
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
    """Generate PDF profile for a student"""
    try:
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
        
        # Add student details
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(0, 0, 0)
        
        # Process extracurricular activities
        ekskul_diikuti = []
        ekskul_cols_full_names = ["Ekstrakurikuler Komputer", "Ekstrakurikuler Pertanian", 
                                 "Ekstrakurikuler Menjahit", "Ekstrakurikuler Pramuka"]
        for col in ekskul_cols_full_names:
            val = data_siswa_dict.get(col)
            if val is not None and (val == 1 or str(val).strip() == '1'):
                ekskul_diikuti.append(col.replace("Ekstrakurikuler ", ""))

        display_data = {
            "Nomor Induk": data_siswa_dict.get("No", "-"),
            "Jenis Kelamin": data_siswa_dict.get("JK", "-"),
            "Kelas": data_siswa_dict.get("Kelas", "-"),
            "Rata-rata Nilai Akademik": f"{data_siswa_dict.get('Rata Rata Nilai Akademik', 0):.2f}",
            "Persentase Kehadiran": f"{data_siswa_dict.get('Kehadiran', 0):.2%}",
            "Ekstrakurikuler yang Diikuti": ", ".join(ekskul_diikuti) if ekskul_diikuti else "Tidak mengikuti ekstrakurikuler",
        }
        
        for key, val in display_data.items():
            pdf.cell(0, 7, f"{key}: {val}", ln=True)
        
        # Return PDF as bytes
        return bytes(pdf.output(dest='S'))
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def preprocess_data(df):
    """Preprocess data for clustering"""
    try:
        df_processed = df.copy()
        df_processed.columns = [col.strip() for col in df_processed.columns]
        
        # Check for missing columns
        missing_cols = [col for col in NUMERIC_COLS + CATEGORICAL_COLS if col not in df_processed.columns]
        if missing_cols:
            st.error(f"Kolom-kolom berikut tidak ditemukan dalam data Anda: {', '.join(missing_cols)}. "
                    "Harap periksa file Excel Anda dan pastikan nama kolom sudah benar.")
            return None, None
        
        df_clean_for_clustering = df_processed.drop(columns=ID_COLS, errors="ignore")
        
        # Handle categorical columns
        for col in CATEGORICAL_COLS:
            df_clean_for_clustering[col] = df_clean_for_clustering[col].fillna(0).astype(str)
        
        # Handle numeric columns
        for col in NUMERIC_COLS:
            if df_clean_for_clustering[col].isnull().any():
                mean_val = df_clean_for_clustering[col].mean()
                df_clean_for_clustering[col] = df_clean_for_clustering[col].fillna(mean_val)
                st.warning(f"Nilai kosong pada kolom '{col}' diisi dengan rata-rata: {mean_val:.2f}.")
        
        # Normalize numeric columns
        scaler = StandardScaler()
        df_clean_for_clustering[NUMERIC_COLS] = scaler.fit_transform(df_clean_for_clustering[NUMERIC_COLS])
        
        return df_clean_for_clustering, scaler
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None, None

def run_kprototypes_clustering(df_preprocessed, n_clusters):
    """Run K-Prototypes clustering"""
    try:
        df_for_clustering = df_preprocessed.copy()
        X_data = df_for_clustering[ALL_FEATURES_FOR_CLUSTERING]
        X = X_data.to_numpy()
        categorical_feature_indices = [X_data.columns.get_loc(c) for c in CATEGORICAL_COLS]
        
        kproto = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=10, verbose=0, random_state=42, n_jobs=1)
        clusters = kproto.fit_predict(X, categorical=categorical_feature_indices)
        
        df_for_clustering["Klaster"] = clusters
        return df_for_clustering, kproto, categorical_feature_indices
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan K-Prototypes: {e}. "
                "Pastikan data Anda cukup bervariasi untuk jumlah klaster yang dipilih.")
        return None, None, None

def generate_cluster_descriptions(df_clustered, n_clusters, numeric_cols, categorical_cols):
    """Generate descriptions for clusters"""
    cluster_characteristics_map = {}
    if 'df_original' not in st.session_state or st.session_state.df_original is None:
        return {}
    
    for i in range(n_clusters):
        cluster_data = df_clustered[df_clustered["Klaster"] == i]
        avg_scaled_values = cluster_data[numeric_cols].mean()
        
        if len(cluster_data) == 0:
            continue
            
        mode_values = cluster_data[categorical_cols].mode().iloc[0] if not cluster_data[categorical_cols].empty else pd.Series()
        
        desc = ""
        
        # Analyze academic performance
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
        
        # Analyze attendance
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
        
        # Analyze extracurricular activities
        if not mode_values.empty:
            ekskul_aktif_modes = [col_name for col_name in categorical_cols if mode_values[col_name] == '1']
            if ekskul_aktif_modes:
                desc += f"Siswa di klaster ini aktif dalam ekstrakurikuler: {', '.join([c.replace('Ekstrakurikuler ', '') for c in ekskul_aktif_modes])}."
            else:
                desc += "Siswa di klaster ini kurang aktif dalam kegiatan ekstrakurikuler."
        
        cluster_characteristics_map[i] = desc
    
    return cluster_characteristics_map

# --- INISIALISASI SESSION STATE ---
def init_session_state():
    """Initialize session state variables"""
    session_vars = {
        'role': None,
        'df_original': None,
        'df_preprocessed_for_clustering': None,
        'df_clustered': None,
        'scaler': None,
        'kproto_model': None,
        'categorical_features_indices': None,
        'n_clusters': 3,
        'cluster_characteristics_map': {},
        'current_menu': None,
        'kepsek_current_menu': "Lihat Hasil Klasterisasi"
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

init_session_state()

# --- FUNGSI HALAMAN UTAMA ---

def show_operator_tu_page():
    """Show Operator TU page"""
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
    
    if st.session_state.current_menu not in menu_options:
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
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Keluar", key="logout_tu_sidebar"):
        st.session_state.clear()
        st.rerun()

    # Menu content
    if st.session_state.current_menu == "Unggah Data":
        show_upload_data_page()
    elif st.session_state.current_menu == "Praproses & Normalisasi Data":
        show_preprocess_data_page()
    elif st.session_state.current_menu == "Klasterisasi Data K-Prototypes":
        show_clustering_page()
    elif st.session_state.current_menu == "Prediksi Klaster Siswa Baru":
        show_prediction_page()
    elif st.session_state.current_menu == "Visualisasi & Profil Klaster":
        show_visualization_page()
    elif st.session_state.current_menu == "Lihat Profil Siswa Individual":
        show_individual_profile_page()

def show_upload_data_page():
    """Show upload data page"""
    st.header("Unggah Data Siswa")
    st.info(
        "Silakan unggah file Excel (.xlsx) yang berisi dataset siswa. "
        "Pastikan file Anda memiliki kolom: No, Nama, JK, Kelas, "
        "Rata Rata Nilai Akademik, Kehadiran, dan kolom ekstrakurikuler."
    )
    
    uploaded_file = st.file_uploader("Pilih File Excel Dataset", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            st.session_state.df_original = df
            st.session_state.df_clustered = None
            st.success("Data berhasil diunggah! Anda dapat melanjutkan ke langkah praproses.")
            st.subheader("Preview Data yang Diunggah:")
            st.dataframe(df, use_container_width=True, height=300)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

def show_preprocess_data_page():
    """Show preprocess data page"""
    st.header("Praproses Data & Normalisasi Z-score")
    
    if st.session_state.df_original is None or st.session_state.df_original.empty:
        st.warning("Silakan unggah data terlebih dahulu di menu 'Unggah Data'.")
        return
    
    st.info(
        "Pada tahap ini, data akan disiapkan untuk analisis klasterisasi dengan "
        "pembersihan data, konversi tipe data, dan normalisasi Z-score."
    )
    
    if st.button("Jalankan Praproses & Normalisasi"):
        with st.spinner("Sedang memproses dan menormalisasi data..."):
            df_preprocessed, scaler = preprocess_data(st.session_state.df_original)
        
        if df_preprocessed is not None and scaler is not None:
            st.session_state.df_preprocessed_for_clustering = df_preprocessed
            st.session_state.scaler = scaler
            st.success("Praproses dan Normalisasi berhasil dilakukan!")
            st.subheader("Data Setelah Praproses dan Normalisasi:")
            st.dataframe(df_preprocessed, use_container_width=True, height=300)

def show_clustering_page():
    """Show clustering page"""
    st.header("Klasterisasi K-Prototypes")
    
    if st.session_state.df_preprocessed_for_clustering is None:
        st.warning("Silakan lakukan praproses data terlebih dahulu.")
        return
    
    st.info(
        "Pilih jumlah klaster (K) untuk mengelompokkan siswa berdasarkan "
        "kombinasi fitur numerik dan kategorikal."
    )
    
    k = st.slider("Pilih Jumlah Klaster (K)", 2, 6, value=st.session_state.n_clusters)
    
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
            
            st.success(f"Klasterisasi selesai dengan {k} klaster!")
            st.subheader("Data Hasil Klasterisasi:")
            st.dataframe(df_final, use_container_width=True, height=300)
            
            # Show cluster summary
            jumlah_per_klaster = df_final["Klaster"].value_counts().sort_index().reset_index()
            jumlah_per_klaster.columns = ["Klaster", "Jumlah Siswa"]
            st.subheader("Ringkasan Klaster:")
            st.table(jumlah_per_klaster)
            
            # Save file for principal
            try:
                df_final_for_kepsek = df_final.copy()
                df_final_for_kepsek['Kehadiran'] = df_final_for_kepsek['Kehadiran'].apply(lambda x: f"{x:.2%}")
                file_name = "Data MA-ALHIKMAH.xlsx"
                df_final_for_kepsek.to_excel(file_name, index=False)
                st.success(f"Hasil disimpan ke file '{file_name}'")
            except Exception as e:
                st.error(f"Gagal menyimpan file Excel: {e}")

def show_prediction_page():
    """Show prediction page for new students"""
    st.header("Prediksi Klaster untuk Siswa Baru")
    
    if st.session_state.kproto_model is None or st.session_state.scaler is None:
        st.warning("Silakan lakukan klasterisasi terlebih dahulu.")
        return
    
    st.info("Masukkan data siswa baru untuk memprediksi klasternya.")
    
    with st.form("form_input_siswa_baru"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Akademik & Kehadiran")
            input_rata_nilai = st.number_input(
                "Rata-rata Nilai Akademik (0-100)", 
                min_value=0.0, max_value=100.0, 
                value=None, format="%.2f"
            )
            input_kehadiran = st.number_input(
                "Persentase Kehadiran (0.0-1.0)", 
                min_value=0.0, max_value=1.0, 
                value=None, format="%.2f"
            )
        
        with col2:
            st.subheader("Keikutsertaan Ekstrakurikuler")
            st.write("Centang ekstrakurikuler yang diikuti:")
            input_cat_ekskul_values = []
            for idx, col in enumerate(CATEGORICAL_COLS):
                val = st.checkbox(col.replace("Ekstrakurikuler ", ""), key=f"ekskul_{idx}")
                input_cat_ekskul_values.append(1 if val else 0)
        
        submitted = st.form_submit_button("Prediksi Klaster Siswa")
    
    if submitted:
        if input_rata_nilai is None or input_kehadiran is None:
            st.error("Harap isi semua nilai numerik terlebih dahulu.")
        else:
            try:
                input_numeric_data = [input_rata_nilai, input_kehadiran]
                normalized_numeric_data = st.session_state.scaler.transform([input_numeric_data])[0]
                
                new_student_data = np.array(
                    list(normalized_numeric_data) + input_cat_ekskul_values, 
                    dtype=object
                ).reshape(1, -1)
                
                predicted_cluster = st.session_state.kproto_model.predict(
                    new_student_data, 
                    categorical=st.session_state.categorical_features_indices
                )
                
                st.success(f"Prediksi Klaster: {predicted_cluster[0]}")
                
                klaster_desc = st.session_state.cluster_characteristics_map.get(
                    predicted_cluster[0], "Deskripsi tidak tersedia."
                )
                st.info(f"Karakteristik Klaster: {klaster_desc}")
                
                # Visualization
                st.subheader("Visualisasi Profil Siswa Baru")
                values_for_plot = list(normalized_numeric_data) + input_cat_ekskul_values
                labels_for_plot = ["Nilai (Norm)", "Kehadiran (Norm)"] + [
                    col.replace("Ekstrakurikuler ", "") for col in CATEGORICAL_COLS
                ]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(labels_for_plot, values_for_plot, color='skyblue')
                ax.set_title("Profil Siswa Baru", fontsize=16, weight='bold')
                ax.set_ylabel("Nilai (Dinormalisasi / Biner)")
                plt.xticks(rotation=45)
                
                for bar, val in zip(bars, values_for_plot):
                    ax.text(bar.get_x() + bar.get_width() / 2, 
                           bar.get_height() + 0.01, 
                           f"{val:.2f}", ha='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()  # Clear figure to prevent memory issues
                
            except Exception as e:
                st.error(f"Error dalam prediksi: {e}")

def show_visualization_page():
    """Show cluster visualization page"""
    st.header("Visualisasi dan Interpretasi Profil Klaster")
    
    if st.session_state.df_preprocessed_for_clustering is None:
        st.warning("Silakan unggah data dan lakukan praproses terlebih dahulu.")
        return
    
    st.info("Pilih jumlah klaster untuk melihat visualisasi karakteristik kelompok.")
    
    k_visual = st.slider("Jumlah Klaster (K)", 2, 6, value=st.session_state.n_clusters)
    
    df_for_visual_clustering, kproto_visual, cat_indices_visual = run_kprototypes_clustering(
        st.session_state.df_preprocessed_for_clustering, k_visual
    )
    
    if df_for_visual_clustering is not None:
        cluster_characteristics_map_visual = generate_cluster_descriptions(
            df_for_visual_clustering, k_visual, NUMERIC_COLS, CATEGORICAL_COLS
        )
        
        st.subheader(f"Profil Klaster untuk K = {k_visual}")
        
        for i in range(k_visual):
            st.markdown("---")
            st.subheader(f"Klaster {i}")
            
            cluster_data = df_for_visual_clustering[df_for_visual_clustering["Klaster"] == i]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Statistik Klaster")
                st.markdown(f"Jumlah Siswa: {len(cluster_data)}")
                
                if not cluster_data.empty:
                    st.write("Rata-rata (Dinormalisasi):")
                    avg_stats = cluster_data[NUMERIC_COLS].mean().round(2)
                    st.dataframe(avg_stats.to_frame(name='Rata-rata'))
                    
                    st.write("Kecenderungan Ekstrakurikuler:")
                    if not cluster_data[CATEGORICAL_COLS].empty:
                        mode_ekskul = cluster_data[CATEGORICAL_COLS].mode()
                        if not mode_ekskul.empty:
                            mode_display = mode_ekskul.iloc[0].apply(lambda x: 'Ya' if x == '1' else 'Tidak')
                            st.dataframe(mode_display.to_frame(name='Paling Umum'))
                
                desc = cluster_characteristics_map_visual.get(i, 'Deskripsi tidak tersedia.')
                st.info(f"Karakteristik: {desc}")
            
            with col2:
                st.markdown("#### Grafik Profil Klaster")
                
                if not cluster_data.empty:
                    try:
                        values_numeric = cluster_data[NUMERIC_COLS].mean().tolist()
                        values_categorical = []
                        
                        for col in CATEGORICAL_COLS:
                            mode_val = cluster_data[col].mode()
                            if not mode_val.empty:
                                values_categorical.append(int(mode_val.iloc[0]))
                            else:
                                values_categorical.append(0)
                        
                        values_for_plot = values_numeric + values_categorical
                        labels_for_plot = ["Nilai", "Kehadiran"] + [
                            col.replace("Ekstrakurikuler ", "") for col in CATEGORICAL_COLS
                        ]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(labels_for_plot, values_for_plot, color='lightcoral')
                        ax.set_title(f"Profil Klaster {i}", fontsize=16, weight='bold')
                        ax.set_ylabel("Nilai (Dinormalisasi / Biner)")
                        plt.xticks(rotation=45)
                        
                        for bar, val in zip(bars, values_for_plot):
                            ax.text(bar.get_x() + bar.get_width() / 2, 
                                   bar.get_height() + 0.01, 
                                   f"{val:.2f}", ha='center', fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.clf()
                        
                    except Exception as e:
                        st.error(f"Error membuat visualisasi: {e}")

def show_individual_profile_page():
    """Show individual student profile page"""
    st.header("Lihat Profil Siswa Berdasarkan Nama")
    
    if st.session_state.df_clustered is None or st.session_state.df_original is None:
        st.warning("Silakan unggah data dan lakukan klasterisasi terlebih dahulu.")
        return
    
    st.info("Pilih nama siswa untuk melihat detail profil dan karakteristik klasternya.")
    
    df_with_cluster = st.session_state.df_clustered
    
    nama_terpilih = st.selectbox(
        "Pilih Nama Siswa",
        df_with_cluster["Nama"].unique(),
        key="pilih_nama_siswa"
    )
    
    if nama_terpilih:
        siswa_data = df_with_cluster[df_with_cluster["Nama"] == nama_terpilih].iloc[0]
        klaster_siswa = siswa_data['Klaster']
        
        st.success(f"Siswa {nama_terpilih} tergolong dalam Klaster {klaster_siswa}")
        
        klaster_desc = st.session_state.cluster_characteristics_map.get(
            klaster_siswa, "Deskripsi tidak tersedia."
        )
        st.info(f"Karakteristik Klaster: {klaster_desc}")
        
        col_info, col_chart = st.columns([1, 2])
        
        with col_info:
            st.markdown("#### Informasi Dasar")
            st.markdown(f"**Nomor Induk:** {siswa_data.get('No', '-')}")
            st.markdown(f"**Jenis Kelamin:** {siswa_data.get('JK', '-')}")
            st.markdown(f"**Kelas:** {siswa_data.get('Kelas', '-')}")
            st.markdown(f"**Nilai Akademik:** {siswa_data.get('Rata Rata Nilai Akademik', 0):.2f}")
            st.markdown(f"**Kehadiran:** {siswa_data.get('Kehadiran', 0):.2%}")
            
            st.markdown("#### Ekstrakurikuler")
            ekskul_diikuti = []
            for col in CATEGORICAL_COLS:
                if siswa_data.get(col, 0) == 1:
                    ekskul_diikuti.append(col.replace("Ekstrakurikuler ", ""))
            
            if ekskul_diikuti:
                for ekskul in ekskul_diikuti:
                    st.markdown(f"- {ekskul} ✅")
            else:
                st.markdown("Tidak mengikuti ekstrakurikuler ❌")
        
        with col_chart:
            st.markdown("#### Visualisasi Profil Siswa")
            
            try:
                labels = ["Nilai Akademik", "Kehadiran (%)"] + [
                    col.replace("Ekstrakurikuler ", "") for col in CATEGORICAL_COLS
                ]
                values = [
                    siswa_data["Rata Rata Nilai Akademik"],
                    siswa_data["Kehadiran"] * 100
                ] + [siswa_data[col] * 100 for col in CATEGORICAL_COLS]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(labels, values, color='lightgreen')
                ax.set_title(f"Profil {nama_terpilih}", fontsize=16, weight='bold')
                ax.set_ylabel("Nilai / Status (%)")
                plt.xticks(rotation=45)
                
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, 
                           bar.get_height() + 1, 
                           f"{val:.1f}", ha='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
                
            except Exception as e:
                st.error(f"Error membuat visualisasi: {e}")
        
        # Show other students in same cluster
        st.subheader(f"Siswa Lain di Klaster {klaster_siswa}")
        siswa_lain = df_with_cluster[
            (df_with_cluster['Klaster'] == klaster_siswa) & 
            (df_with_cluster['Nama'] != nama_terpilih)
        ]
        
        if not siswa_lain.empty:
            display_cols = ["No", "Nama", "JK", "Kelas", "Rata Rata Nilai Akademik", "Kehadiran"]
            display_df = siswa_lain[display_cols].copy()
            display_df["Kehadiran"] = display_df["Kehadiran"].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("Tidak ada siswa lain dalam klaster ini.")
        
        # PDF generation
        st.subheader("Unduh Laporan PDF")
        if st.session_state.cluster_characteristics_map:
            if st.button("Generate & Unduh PDF"):
                with st.spinner("Menyiapkan PDF..."):
                    siswa_data_for_pdf = siswa_data.drop(labels=["Klaster"]).to_dict()
                    pdf_bytes = generate_pdf_profil_siswa(
                        nama_terpilih,
                        siswa_data_for_pdf,
                        siswa_data["Klaster"],
                        st.session_state.cluster_characteristics_map
                    )
                
                if pdf_bytes:
                    st.success("PDF berhasil disiapkan!")
                    st.download_button(
                        label="Unduh PDF",
                        data=pdf_bytes,
                        file_name=f"Profil_{nama_terpilih.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )

def show_kepala_sekolah_page():
    """Show principal page"""
    file_path = "Data MA-ALHIKMAH.xlsx"
    
    # Load data if exists
    if os.path.exists(file_path):
        try:
            df_kepsek_load = pd.read_excel(file_path, engine='openpyxl')
            st.session_state.df_clustered = df_kepsek_load
            
            if st.session_state.df_original is None:
                df_original_from_clustered = df_kepsek_load.copy()
                if df_original_from_clustered['Kehadiran'].dtype == 'object':
                    df_original_from_clustered['Kehadiran'] = (
                        df_original_from_clustered['Kehadiran']
                        .str.rstrip('%').astype('float') / 100
                    )
                st.session_state.df_original = df_original_from_clustered.drop(
                    columns=['Klaster'], errors='ignore'
                )
                
                n_clusters_kepsek = len(df_kepsek_load['Klaster'].unique())
                st.session_state.n_clusters = n_clusters_kepsek
                
                df_preprocessed, _ = preprocess_data(st.session_state.df_original)
                if df_preprocessed is not None:
                    df_preprocessed['Klaster'] = df_kepsek_load['Klaster']
                    st.session_state.cluster_characteristics_map = generate_cluster_descriptions(
                        df_preprocessed, n_clusters_kepsek, NUMERIC_COLS, CATEGORICAL_COLS
                    )
        except Exception as e:
            st.error(f"Error membaca file: {e}")
            st.session_state.df_clustered = None
    
    # Sidebar navigation
    st.sidebar.title("MENU NAVIGASI")
    st.sidebar.markdown("---")
    
    kepsek_menu_options = [
        "Lihat Hasil Klasterisasi",
        "Visualisasi & Profil Klaster", 
        "Lihat Profil Siswa Individual"
    ]
    
    for option in kepsek_menu_options:
        icon_map = {
            "Lihat Hasil Klasterisasi": "📋",
            "Visualisasi & Profil Klaster": "📈",
            "Lihat Profil Siswa Individual": "👤"
        }
        display_name = f"{icon_map.get(option, '')} {option}"
        button_key = f"kepsek_{option.replace(' ', '_')}"
        
        if st.sidebar.button(display_name, key=button_key):
            st.session_state.kepsek_current_menu = option
            st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Keluar", key="logout_kepsek"):
        st.session_state.clear()
        st.rerun()
    
    st.title("👨‍💼 Dasbor Kepala Sekolah")
    
    if st.session_state.df_clustered is None or st.session_state.df_clustered.empty:
        st.warning(
            "File hasil klasterisasi tidak ditemukan. "
            "Mohon minta Operator TU untuk memproses data terlebih dahulu."
        )
        return
    
    # Show content based on selected menu
    if st.session_state.kepsek_current_menu == "Lihat Hasil Klasterisasi":
        show_kepsek_clustering_results()
    elif st.session_state.kepsek_current_menu == "Visualisasi & Profil Klaster":
        show_kepsek_visualization()
    elif st.session_state.kepsek_current_menu == "Lihat Profil Siswa Individual":
        show_kepsek_individual_profile()

def show_kepsek_clustering_results():
    """Show clustering results for principal"""
    st.header("Hasil Klasterisasi Siswa")
    st.info("Data siswa yang sudah dikelompokkan ke dalam klaster.")
    
    st.subheader("Data Hasil Klasterisasi")
    st.dataframe(st.session_state.df_clustered, use_container_width=True, height=300)
    
    st.subheader("Ringkasan Klaster")
    jumlah_per_klaster = (st.session_state.df_clustered["Klaster"]
                         .value_counts().sort_index().reset_index())
    jumlah_per_klaster.columns = ["Klaster", "Jumlah Siswa"]
    st.table(jumlah_per_klaster)

def show_kepsek_visualization():
    """Show visualization for principal"""
    st.header("Visualisasi dan Interpretasi Profil Klaster")
    st.info("Visualisasi dan ringkasan karakteristik setiap kelompok siswa.")
    
    if not st.session_state.cluster_characteristics_map:
        st.warning("Deskripsi klaster tidak tersedia.")
        return
    
    st.subheader(f"Karakteristik Klaster ({st.session_state.n_clusters} Klaster)")
    
    for i in range(st.session_state.n_clusters):
        with st.expander(f"Klaster {i}"):
            cluster_data = st.session_state.df_clustered[
                st.session_state.df_clustered["Klaster"] == i
            ]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write(f"**Jumlah Siswa:** {len(cluster_data)}")
                desc = st.session_state.cluster_characteristics_map.get(
                    i, 'Deskripsi tidak tersedia.'
                )
                st.write(f"**Karakteristik:** {desc}")
            
            with col2:
                if not cluster_data.empty:
                    avg_nilai = cluster_data["Rata Rata Nilai Akademik"].mean()
                    avg_kehadiran = cluster_data["Kehadiran"].mean()
                    
                    # Handle percentage format in kehadiran
                    if isinstance(avg_kehadiran, str):
                        avg_kehadiran = float(avg_kehadiran.replace('%', '')) / 100
                    
                    st.write(f"**Rata-rata Nilai:** {avg_nilai:.2f}")
                    st.write(f"**Rata-rata Kehadiran:** {avg_kehadiran:.2%}")

def show_kepsek_individual_profile():
    """Show individual profile for principal"""
    st.header("Profil Siswa Individual")
    st.info("Pilih siswa untuk melihat profil dan karakteristik klasternya.")
    
    df_kepsek = st.session_state.df_clustered
    
    nama_terpilih = st.selectbox(
        "Pilih Nama Siswa",
        df_kepsek["Nama"].unique(),
        key="kepsek_pilih_siswa"
    )
    
    if nama_terpilih:
        siswa_data = df_kepsek[df_kepsek["Nama"] == nama_terpilih].iloc[0]
        klaster_siswa = siswa_data['Klaster']
        
        st.success(f"Siswa {nama_terpilih} tergolong dalam Klaster {klaster_siswa}")
        
        klaster_desc = st.session_state.cluster_characteristics_map.get(
            klaster_siswa, "Deskripsi tidak tersedia."
        )
        st.info(f"Karakteristik Klaster: {klaster_desc}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Informasi Siswa")
            st.write(f"**Nomor:** {siswa_data.get('No', '-')}")
            st.write(f"**Jenis Kelamin:** {siswa_data.get('JK', '-')}")
            st.write(f"**Kelas:** {siswa_data.get('Kelas', '-')}")
            st.write(f"**Nilai Akademik:** {siswa_data.get('Rata Rata Nilai Akademik', 0):.2f}")
            
            # Handle kehadiran format
            kehadiran = siswa_data.get('Kehadiran', 0)
            if isinstance(kehadiran, str):
                st.write(f"**Kehadiran:** {kehadiran}")
            else:
                st.write(f"**Kehadiran:** {kehadiran:.2%}")
        
        with col2:
            st.subheader("Ekstrakurikuler")
            ekskul_diikuti = []
            for col in CATEGORICAL_COLS:
                val = siswa_data.get(col, 0)
                if val == 1 or str(val) == '1':
                    ekskul_diikuti.append(col.replace("Ekstrakurikuler ", ""))
            
            if ekskul_diikuti:
                for ekskul in ekskul_diikuti:
                    st.write(f"✅ {ekskul}")
            else:
                st.write("❌ Tidak mengikuti ekstrakurikuler")
        
        # PDF download
        st.subheader("Unduh Laporan PDF")
        if st.button("Generate PDF", key="kepsek_pdf"):
            with st.spinner("Menyiapkan PDF..."):
                siswa_data_for_pdf = siswa_data.drop(labels=["Klaster"]).to_dict()
                
                # Convert percentage string back to float for PDF
                if isinstance(siswa_data_for_pdf.get('Kehadiran'), str):
                    kehadiran_str = siswa_data_for_pdf['Kehadiran'].replace('%', '')
                    siswa_data_for_pdf['Kehadiran'] = float(kehadiran_str) / 100
                
                pdf_bytes = generate_pdf_profil_siswa(
                    nama_terpilih,
                    siswa_data_for_pdf,
                    siswa_data["Klaster"],
                    st.session_state.cluster_characteristics_map
                )
            
            if pdf_bytes:
                st.success("PDF berhasil disiapkan!")
                st.download_button(
                    label="Unduh PDF",
                    data=pdf_bytes,
                    file_name=f"Profil_{nama_terpilih.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    key="kepsek_download_pdf"
                )

# --- MAIN APPLICATION LOGIC ---

def main():
    """Main application function"""
    if st.session_state.role is None:
        # Login page
        st.sidebar.empty()
        
        st.markdown("""
        <div class="login-container">
            <div class="login-card">
                <h2>Pilih Peran Anda</h2>
                <p style='margin-bottom: 25px;'>
                    Selamat datang di sistem pengelompokan siswa. 
                    Silakan pilih peran Anda untuk melanjutkan.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Masuk sebagai **Operator TU**", use_container_width=True):
                st.session_state.role = 'Operator TU'
                st.session_state.current_menu = "Unggah Data"
                st.rerun()
        
        with col2:
            if st.button("Masuk sebagai **Kepala Sekolah**", use_container_width=True):
                st.session_state.role = 'Kepala Sekolah'
                st.session_state.kepsek_current_menu = "Lihat Hasil Klasterisasi"
                st.rerun()
    
    elif st.session_state.role == 'Operator TU':
        show_operator_tu_page()
    
    elif st.session_state.role == 'Kepala Sekolah':
        show_kepala_sekolah_page()

if __name__ == "__main__":
    main()
