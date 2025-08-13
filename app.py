import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import io

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
    .st-sidebar-button-active {{
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
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 999;
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
    .stForm {{
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 25px !important;
        margin-bottom: 25px !important;
        border: 1px solid #e0e0e0;
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
    for col in CATEGORICAL_COLS:
        val = data_siswa_dict.get(col)
        if val is not None and (val == 1 or str(val).strip() == '1'):
            ekskul_diikuti.append(col.replace("Ekstrakurikuler ", ""))

    kehadiran_val = data_siswa_dict.get('Kehadiran', 0)
    nilai_akademik_val = data_siswa_dict.get('Rata Rata Nilai Akademik', 0)

    display_data = {
        "Nomor Induk": data_siswa_dict.get("No", "-"),
        "Jenis Kelamin": data_siswa_dict.get("JK", "-"),
        "Kelas": data_siswa_dict.get("Kelas", "-"),
        "Rata-rata Nilai Akademik": f"{nilai_akademik_val:.2f}",
        "Persentase Kehadiran": f"{kehadiran_val:.2%}",
        "Ekstrakurikuler yang Diikuti": ", ".join(ekskul_diikuti) if ekskul_diikuti else "Tidak mengikuti ekstrakurikuler",
    }
    for key, val in display_data.items():
        pdf.cell(0, 7, f"{key}: {val}", ln=True)
    
    try:
        # Menghasilkan PDF sebagai byte string
        pdf_output = pdf.output(dest='S').encode('latin-1')
        return bytes(pdf_output)
    except Exception as e:
        st.error(f"Error saat mengonversi PDF: {e}. Coba pastikan tidak ada karakter aneh pada data.")
        return None


def preprocess_data(df):
    df_processed = df.copy()
    df_processed.columns = [col.strip() for col in df_processed.columns]
    
    missing_cols = [col for col in ALL_FEATURES_FOR_CLUSTERING if col not in df_processed.columns]
    if missing_cols:
        st.error(f"Kolom-kolom berikut tidak ditemukan: {', '.join(missing_cols)}. Periksa file Excel Anda.")
        return None, None
    
    # Memilih hanya kolom yang akan digunakan untuk clustering
    df_clean_for_clustering = df_processed[ALL_FEATURES_FOR_CLUSTERING].copy()

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
    X_data = df_preprocessed[ALL_FEATURES_FOR_CLUSTERING]
    X = X_data.to_numpy()
    categorical_feature_indices = [X_data.columns.get_loc(c) for c in CATEGORICAL_COLS]
    
    try:
        kproto = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=10, verbose=0, random_state=42, n_jobs=1) # n_jobs=1 lebih aman untuk deployment
        clusters = kproto.fit_predict(X, categorical=categorical_feature_indices)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan K-Prototypes: {e}. Pastikan data cukup bervariasi.")
        return None, None, None
        
    df_clustered = df_preprocessed.copy()
    df_clustered["Klaster"] = clusters
    return df_clustered, kproto, categorical_feature_indices

def generate_cluster_descriptions(df_clustered_normalized, n_clusters):
    cluster_characteristics_map = {}
    for i in range(n_clusters):
        cluster_data = df_clustered_normalized[df_clustered_normalized["Klaster"] == i]
        if cluster_data.empty:
            continue

        avg_scaled_values = cluster_data[NUMERIC_COLS].mean()
        mode_values = cluster_data[CATEGORICAL_COLS].mode().iloc[0]
        
        desc = ""
        # Deskripsi Nilai Akademik
        if avg_scaled_values["Rata Rata Nilai Akademik"] > 0.75: desc += "Nilai akademik sangat tinggi. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] > 0.25: desc += "Nilai akademik di atas rata-rata. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] < -0.75: desc += "Nilai akademik sangat rendah. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] < -0.25: desc += "Nilai akademik di bawah rata-rata. "
        else: desc += "Nilai akademik rata-rata. "

        # Deskripsi Kehadiran
        if avg_scaled_values["Kehadiran"] > 0.75: desc += "Kehadiran sangat tinggi. "
        elif avg_scaled_values["Kehadiran"] > 0.25: desc += "Kehadiran di atas rata-rata. "
        elif avg_scaled_values["Kehadiran"] < -0.75: desc += "Kehadiran sangat rendah. "
        elif avg_scaled_values["Kehadiran"] < -0.25: desc += "Kehadiran di bawah rata-rata. "
        else: desc += "Kehadiran rata-rata. "
        
        ekskul_aktif_modes = [col for col in CATEGORICAL_COLS if mode_values[col] == '1']
        if ekskul_aktif_modes:
            desc += f"Aktif di ekstrakurikuler: {', '.join([c.replace('Ekstrakurikuler ', '') for c in ekskul_aktif_modes])}."
        else:
            desc += "Cenderung tidak aktif di ekstrakurikuler."
            
        cluster_characteristics_map[i] = desc
    return cluster_characteristics_map

# --- INISIALISASI SESSION STATE ---
def init_session_state():
    defaults = {
        'role': None,
        'df_original': None,
        'df_preprocessed_for_clustering': None,
        'df_clustered': None,
        'scaler': None,
        'kproto_model': None,
        'categorical_features_indices': None,
        'n_clusters': 3,
        'cluster_characteristics_map': {},
        'current_menu': "Unggah Data",
        'kepsek_current_menu': "Lihat Hasil Klasterisasi"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- FUNGSI HALAMAN (VIEWS) ---

def show_operator_tu_page():
    st.sidebar.title("MENU NAVIGASI OPERATOR")
    st.sidebar.markdown("---")
    
    menu_options = [
        "Unggah Data", "Praproses & Normalisasi Data", "Klasterisasi Data",
        "Prediksi Siswa Baru", "Visualisasi & Profil Klaster", "Profil Siswa Individual"
    ]
    icon_map = {
        "Unggah Data": "⬆", "Praproses & Normalisasi Data": "⚙️", "Klasterisasi Data": "📊",
        "Prediksi Siswa Baru": "🔮", "Visualisasi & Profil Klaster": "📈", "Profil Siswa Individual": "👤"
    }
    
    for option in menu_options:
        if st.sidebar.button(f"{icon_map.get(option, '')} {option}", key=f"nav_{option}"):
            st.session_state.current_menu = option
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Keluar", key="logout_tu"):
        init_session_state() # Reset state
        st.rerun()

    # == KONTEN HALAMAN BERDASARKAN MENU ==
    if st.session_state.current_menu == "Unggah Data":
        st.header("1. Unggah Data Siswa")
        st.info("Unggah file Excel (.xlsx) berisi data siswa. Pastikan nama kolom sesuai template.")
        uploaded_file = st.file_uploader("Pilih File Excel", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                st.session_state.df_original = df
                # Reset state selanjutnya jika data baru diunggah
                st.session_state.df_preprocessed_for_clustering = None
                st.session_state.df_clustered = None
                st.success("Data berhasil diunggah! Lihat preview di bawah.")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    elif st.session_state.current_menu == "Praproses & Normalisasi Data":
        st.header("2. Praproses dan Normalisasi Data")
        if st.session_state.df_original is None:
            st.warning("Silakan unggah data di menu 'Unggah Data' terlebih dahulu.")
        else:
            st.info("Klik tombol di bawah untuk membersihkan dan menormalisasi data agar siap dianalisis.")
            if st.button("Jalankan Praproses"):
                with st.spinner("Memproses data..."):
                    df_preprocessed, scaler = preprocess_data(st.session_state.df_original)
                    if df_preprocessed is not None:
                        st.session_state.df_preprocessed_for_clustering = df_preprocessed
                        st.session_state.scaler = scaler
                        st.success("Praproses dan normalisasi selesai!")
                        st.subheader("Data Setelah Praproses (Fitur untuk Klasterisasi):")
                        st.dataframe(df_preprocessed.head(), use_container_width=True)

    elif st.session_state.current_menu == "Klasterisasi Data":
        st.header("3. Jalankan Klasterisasi K-Prototypes")
        if st.session_state.df_preprocessed_for_clustering is None:
            st.warning("Jalankan praproses data di menu sebelumnya terlebih dahulu.")
        else:
            st.info("Pilih jumlah klaster (K) yang diinginkan, lalu jalankan algoritma.")
            k = st.slider("Pilih Jumlah Klaster (K)", 2, 6, st.session_state.n_clusters)
            if st.button("Jalankan Klasterisasi"):
                with st.spinner(f"Mengelompokkan data ke dalam {k} klaster..."):
                    df_clustered_normalized, kproto_model, cat_indices = run_kprototypes_clustering(
                        st.session_state.df_preprocessed_for_clustering, k
                    )
                    if df_clustered_normalized is not None:
                        df_final = st.session_state.df_original.copy()
                        df_final['Klaster'] = df_clustered_normalized['Klaster']
                        
                        st.session_state.df_clustered = df_final
                        st.session_state.kproto_model = kproto_model
                        st.session_state.categorical_features_indices = cat_indices
                        st.session_state.n_clusters = k
                        st.session_state.cluster_characteristics_map = generate_cluster_descriptions(
                            df_clustered_normalized, k
                        )
                        st.success(f"Klasterisasi dengan {k} klaster selesai! Data siap dilihat oleh Kepala Sekolah.")
                        st.subheader("Data Hasil Klasterisasi:")
                        st.dataframe(df_final, use_container_width=True)

    # Lanjutan fungsi lain (Prediksi, Visualisasi, Profil Individual) ditempatkan di sini...
    elif st.session_state.current_menu == "Prediksi Siswa Baru":
        st.header("4. Prediksi Klaster Siswa Baru")
        if st.session_state.kproto_model is None:
            st.warning("Lakukan klasterisasi di menu 'Klasterisasi Data' untuk melatih model terlebih dahulu.")
        else:
            st.info("Masukkan data siswa baru untuk memprediksi klasternya.")
            with st.form("form_prediksi"):
                col1, col2 = st.columns(2)
                with col1:
                    input_rata_nilai = st.number_input("Rata-rata Nilai Akademik (0-100)", 0.0, 100.0, 75.0)
                    input_kehadiran = st.number_input("Persentase Kehadiran (0-1)", 0.0, 1.0, 0.95, format="%.2f")
                with col2:
                    st.write("Keikutsertaan Ekstrakurikuler:")
                    input_ekskul = [st.checkbox(col.replace("Ekstrakurikuler ", ""), key=f"pred_{col}") for col in CATEGORICAL_COLS]
                
                submitted = st.form_submit_button("Prediksi Klaster")
                if submitted:
                    input_numeric = [input_rata_nilai, input_kehadiran]
                    normalized_numeric = st.session_state.scaler.transform([input_numeric])[0]
                    input_cat = [1 if val else 0 for val in input_ekskul]
                    
                    new_student_data = np.array(list(normalized_numeric) + input_cat, dtype=object).reshape(1, -1)
                    
                    predicted_cluster = st.session_state.kproto_model.predict(
                        new_student_data, categorical=st.session_state.categorical_features_indices
                    )
                    pred_cluster_num = predicted_cluster[0]

                    st.success(f"Hasil Prediksi: Siswa ini masuk ke **Klaster {pred_cluster_num}**")
                    desc = st.session_state.cluster_characteristics_map.get(pred_cluster_num, "Deskripsi tidak ada.")
                    st.markdown(f"**Karakteristik Klaster:** *{desc}*")
    
    elif st.session_state.current_menu == "Visualisasi & Profil Klaster":
        st.header("5. Visualisasi & Profil Klaster")
        if st.session_state.df_clustered is None:
            st.warning("Jalankan klasterisasi di menu 'Klasterisasi Data' terlebih dahulu.")
        else:
            st.info("Berikut adalah visualisasi profil untuk setiap klaster yang terbentuk.")
            for i in range(st.session_state.n_clusters):
                st.markdown(f"---")
                st.subheader(f"Klaster {i}")
                
                cluster_data_original = st.session_state.df_clustered[st.session_state.df_clustered["Klaster"] == i]
                
                # Perlu data ternormalisasi untuk visualisasi yang konsisten
                df_normalized_temp, _ = preprocess_data(st.session_state.df_original)
                df_normalized_temp['Klaster'] = st.session_state.df_clustered['Klaster']
                cluster_data_normalized = df_normalized_temp[df_normalized_temp["Klaster"] == i]

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Jumlah Siswa", len(cluster_data_original))
                    desc = st.session_state.cluster_characteristics_map.get(i, "")
                    st.markdown(f"**Ringkasan:** *{desc}*")

                with col2:
                    values_numeric = cluster_data_normalized[NUMERIC_COLS].mean().tolist()
                    values_ekskul = [int(cluster_data_normalized[col].mode().iloc[0]) for col in CATEGORICAL_COLS]
                    values_plot = values_numeric + values_ekskul
                    labels_plot = ["Nilai (Norm)", "Hadir (Norm)"] + [c.replace("Ekstrakurikuler ", "Ekskul\n") for c in CATEGORICAL_COLS]
                    
                    fig, ax = plt.subplots()
                    sns.barplot(x=labels_plot, y=values_plot, ax=ax, palette="viridis")
                    ax.set_ylabel("Rata-rata (Ternormalisasi / Biner)")
                    ax.set_title(f"Profil Rata-rata Klaster {i}")
                    st.pyplot(fig)

    elif st.session_state.current_menu == "Profil Siswa Individual":
        st.header("6. Lihat Profil Siswa Individual")
        if st.session_state.df_clustered is None:
            st.warning("Jalankan klasterisasi di menu 'Klasterisasi Data' terlebih dahulu.")
        else:
            df_display = st.session_state.df_clustered
            nama_terpilih = st.selectbox("Pilih Nama Siswa", df_display["Nama"].unique())
            if nama_terpilih:
                siswa_data = df_display[df_display["Nama"] == nama_terpilih].iloc[0]
                klaster_siswa = siswa_data['Klaster']
                
                st.success(f"Siswa **{nama_terpilih}** berada di **Klaster {klaster_siswa}**.")
                desc = st.session_state.cluster_characteristics_map.get(klaster_siswa, "")
                st.markdown(f"**Karakteristik Klaster:** *{desc}*")
                st.markdown("---")
                
                st.subheader("Detail Siswa")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Kelas:** {siswa_data['Kelas']}")
                    st.write(f"**Jenis Kelamin:** {siswa_data['JK']}")
                    st.write(f"**Rata-rata Nilai:** {siswa_data['Rata Rata Nilai Akademik']:.2f}")
                    st.write(f"**Kehadiran:** {siswa_data['Kehadiran']:.2%}")
                    
                    # PDF Download
                    pdf_bytes = generate_pdf_profil_siswa(
                        nama_terpilih, 
                        siswa_data.to_dict(), 
                        klaster_siswa, 
                        st.session_state.cluster_characteristics_map
                    )
                    st.download_button(
                        label="📄 Unduh Profil PDF",
                        data=pdf_bytes,
                        file_name=f"Profil_{nama_terpilih.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )

                with col2:
                    st.subheader("Siswa Lain di Klaster yang Sama")
                    siswa_lain = df_display[
                        (df_display['Klaster'] == klaster_siswa) & 
                        (df_display['Nama'] != nama_terpilih)
                    ][["Nama", "Kelas"]]
                    if not siswa_lain.empty:
                        st.dataframe(siswa_lain, use_container_width=True)
                    else:
                        st.info("Tidak ada siswa lain di klaster ini.")

def show_kepala_sekolah_page():
    st.sidebar.title("MENU NAVIGASI KEPSEK")
    st.sidebar.markdown("---")
    
    kepsek_menu_options = ["Lihat Hasil Klasterisasi", "Visualisasi & Profil Klaster", "Lihat Profil Siswa"]
    icon_map = {"Lihat Hasil Klasterisasi": "📋", "Visualisasi & Profil Klaster": "📈", "Lihat Profil Siswa": "👤"}
    
    for option in kepsek_menu_options:
        if st.sidebar.button(f"{icon_map.get(option, '')} {option}", key=f"nav_kepsek_{option}"):
            st.session_state.kepsek_current_menu = option
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Keluar", key="logout_kepsek"):
        init_session_state() # Reset state
        st.rerun()

    # == KONTEN HALAMAN BERDASARKAN MENU ==
    st.title("👨‍💼 Dasbor Kepala Sekolah")
    
    if st.session_state.df_clustered is None or st.session_state.df_clustered.empty:
        st.warning("Data klasterisasi belum tersedia. Mohon minta Operator TU untuk memproses data terlebih dahulu.")
        return

    if st.session_state.kepsek_current_menu == "Lihat Hasil Klasterisasi":
        st.header("Hasil Klasterisasi Siswa")
        st.info("Tabel di bawah ini adalah data siswa yang telah dikelompokkan.")
        st.dataframe(st.session_state.df_clustered, use_container_width=True)
        
        st.subheader("Jumlah Siswa per Klaster")
        cluster_counts = st.session_state.df_clustered["Klaster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

    elif st.session_state.kepsek_current_menu == "Visualisasi & Profil Klaster":
        st.header("Visualisasi dan Interpretasi Profil Klaster")
        st.info("Visualisasi ini membantu memahami karakteristik utama setiap kelompok siswa.")
        
        # Perlu data ternormalisasi untuk visualisasi yang konsisten
        df_normalized_temp, _ = preprocess_data(st.session_state.df_original)
        if df_normalized_temp is None:
             st.error("Gagal memproses ulang data untuk visualisasi.")
             return
             
        df_normalized_temp['Klaster'] = st.session_state.df_clustered['Klaster']

        for i in range(st.session_state.n_clusters):
            st.markdown(f"---")
            st.subheader(f"Klaster {i}")
            
            cluster_data_original = st.session_state.df_clustered[st.session_state.df_clustered["Klaster"] == i]
            cluster_data_normalized = df_normalized_temp[df_normalized_temp["Klaster"] == i]

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Jumlah Siswa", len(cluster_data_original))
                desc = st.session_state.cluster_characteristics_map.get(i, "Deskripsi tidak tersedia.")
                st.markdown(f"**Ringkasan Karakteristik:**")
                st.info(f"{desc}")

            with col2:
                values_numeric = cluster_data_normalized[NUMERIC_COLS].mean().tolist()
                values_ekskul = [int(cluster_data_normalized[col].mode().iloc[0]) for col in CATEGORICAL_COLS]
                values_plot = values_numeric + values_ekskul
                labels_plot = ["Nilai (Norm)", "Hadir (Norm)"] + [c.replace("Ekstrakurikuler ", "Ekskul\n") for c in CATEGORICAL_COLS]
                
                fig, ax = plt.subplots()
                sns.barplot(x=labels_plot, y=values_plot, ax=ax, palette="plasma")
                ax.set_ylabel("Rata-rata (Ternormalisasi / Biner)")
                ax.set_title(f"Profil Rata-rata Klaster {i}")
                plt.tight_layout()
                st.pyplot(fig)

    elif st.session_state.kepsek_current_menu == "Lihat Profil Siswa":
        st.header("Lihat Profil Siswa Individual")
        df_display = st.session_state.df_clustered
        
        nama_terpilih = st.selectbox("Pilih Nama Siswa", df_display["Nama"].unique(), key="kepsek_select_student")
        if nama_terpilih:
            siswa_data = df_display[df_display["Nama"] == nama_terpilih].iloc[0]
            klaster_siswa = siswa_data['Klaster']
            
            st.success(f"Siswa **{nama_terpilih}** berada di **Klaster {klaster_siswa}**.")
            desc = st.session_state.cluster_characteristics_map.get(klaster_siswa, "")
            st.markdown(f"**Karakteristik Klaster:** *{desc}*")
            st.markdown("---")
            
            st.subheader("Detail Siswa")
            st.write(f"**Kelas:** {siswa_data['Kelas']}")
            st.write(f"**Jenis Kelamin:** {siswa_data['JK']}")
            st.write(f"**Rata-rata Nilai:** {siswa_data['Rata Rata Nilai Akademik']:.2f}")
            st.write(f"**Kehadiran:** {siswa_data['Kehadiran']:.2%}")

            ekskul_diikuti = [col.replace("Ekstrakurikuler ", "") for col in CATEGORICAL_COLS if siswa_data[col] == 1]
            if ekskul_diikuti:
                st.write(f"**Ekstrakurikuler:** {', '.join(ekskul_diikuti)}")
            else:
                st.write("**Ekstrakurikuler:** Tidak mengikuti")


def show_login_page():
    st.markdown("""
        <div class='login-container'>
            <div class='login-card'>
                <h2>Selamat Datang</h2>
                <p>Silakan pilih peran Anda untuk melanjutkan</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Saya Operator TU ⚙️", use_container_width=True):
            st.session_state.role = 'operator_tu'
            st.rerun()
    with col2:
        if st.button("Saya Kepala Sekolah 👨‍💼", use_container_width=True):
            st.session_state.role = 'kepala_sekolah'
            st.rerun()
            
    st.markdown("</div></div>", unsafe_allow_html=True)


# --- BLOK EKSEKUSI UTAMA ---
def main():
    init_session_state()
    
    if st.session_state.role is None:
        show_login_page()
    elif st.session_state.role == 'operator_tu':
        show_operator_tu_page()
    elif st.session_state.role == 'kepala_sekolah':
        show_kepala_sekolah_page()

if __name__ == "__main__":
    main()
