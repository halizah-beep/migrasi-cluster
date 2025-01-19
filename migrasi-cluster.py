import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

dfmgr = pd.read_csv('migrasi_kota_bekasi.csv')
X = pd.read_csv('output_cluster.csv')

# Konfigurasi awal
st.set_page_config(
    page_title="Clustering Migrasi Penduduk",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inisialisasi session_state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'X_normalized' not in st.session_state:
    st.session_state['X_normalized'] = None
if 'kmeans' not in st.session_state:
    st.session_state['kmeans'] = None
if 'num_clusters' not in st.session_state:
    st.session_state['num_clusters'] = None

# Tambahkan CSS kustom
st.markdown(
    """
    <style>
    /* Style dasar untuk tombol di sidebar */
    .stButton > button {
        width: 100%;                  /* Samakan lebar tombol */
        background-color: #f0f0f0;    /* Warna latar tombol (ungu) */
        border: 2px solid #6a0dad;    /* Warna border tombol (ungu) */
        color: #6a0dad;                 /* Warna teks */
        font-weight: bold;            /* Tebalkan teks */
        margin: 5px 0;                /* Jarak antar tombol */
        transition: 0.3s;             /* Efek transisi hover */
        border-radius: 5px;           /* Sudut tombol melengkung */
    }
    /* Hover untuk tombol */
    .stButton > button:hover {
        background-color: #a32cc4;    /* Warna latar saat hover (ungu terang) */
        color: white;                 /* Warna teks tetap putih */
    }
    /* Tombol aktif */
    .active > button {
        background-color: #4b0082 !important; /* Warna latar saat aktif (ungu gelap) */
        color: white !important;              /* Warna teks tetap putih */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Fungsi untuk navigasi
def set_menu(menu_name) :
    st.session_state['menu'] = menu_name

# Inisialisasi session state untuk menu
if 'menu' not in st.session_state :
    st.session_state['menu'] = "Deskripsi"

# Sidebar 
with st.sidebar :
    st.markdown("## Menu Aplikasi")
    if st.button("Home") :
        set_menu("Home")
    if st.button("Deskripsi") :
        set_menu("Deskripsi")
    if st.button("Unggah Data") :
        set_menu("Unggah Data")
    if st.button("Preprocessing") :
        set_menu("Preprocessing")
    if st.button("Clustering") :
        set_menu("Clustering")
    if st.button("Visualisasi") :
        set_menu("Visualisasi")
    if st.button("Download Hasil") :
        set_menu("Download Hasil")

# Konten berdasarkan menu
menu = st.session_state['menu']

# Judul aplikasi
st.title("Aplikasi Pemetaan Migrasi Penduduk Menggunakan Clustering K-Means")

# Halaman Home
if menu == "Home":
    st.header("Migrasi Penduduk Kota Bekasi Tahun 2022 - 2023")

    st.image("C:/Users/ARBAI KYB/Downloads/clustering_migrasi/migrasi.jpg", use_container_width=True)
    
    st.subheader("Pengertian Migrasi")
    st.markdown("""
    Migrasi adalah perpindahan penduduk dari satu wilayah ke wilayah lain dengan tujuan menetap. 
    Terdapat dua jenis migrasi utama, yaitu :
    - **Migrasi Masuk (In-migration)** : Penduduk pindah ke suatu wilayah.
    - **Migrasi Keluar (Out-migration)** : Penduduk pindah keluar dari suatu wilayah.
    
    Dalam konteks Kota Bekasi, migrasi dapat memberikan wawasan tentang dinamika kependudukan, seperti pertumbuhan populasi, tekanan sosial, dan kebutuhan infrastruktur.
    """)

    st.subheader("Pilih Kelurahan untuk Melihat Data")
    if not dfmgr.empty:
        col1, col2 = st.columns([1, 3])

        with col1:
            kelurahan_pilihan = st.selectbox(
                "Pilih kelurahan :",
                options=dfmgr['nama_desa_kelurahan'].unique(),
                index=0
            )

        with col2:
            data_kelurahan = dfmgr[dfmgr['nama_desa_kelurahan'] == kelurahan_pilihan]
            st.write(f"Data Migrasi untuk Kelurahan: **{kelurahan_pilihan}**")
            st.write(data_kelurahan)
    
            fig, ax = plt.subplots(figsize=(5, 2))
            sns.heatmap(
                data_kelurahan.iloc[:, 1:],  # Kolom numerik migrasi
                annot=True,
                fmt="d",
                cmap="coolwarm",
                cbar_kws={"label": "Jumlah Migrasi"}
            )
            ax.set_title(f"Distribusi Migrasi - {kelurahan_pilihan}")
            st.pyplot(fig)
    else:
        st.error("Dataset migrasi tidak ditemukan.")
    
    st.subheader("Dataset Migrasi Kota Bekasi (2022 - 2023)")
    st.write(dfmgr)

    if dfmgr is not None:  # Memastikan data tersedia
            dfmgr.set_index('nama_desa_kelurahan', inplace=True)
            if not all(dfmgr.dtypes == 'int64') and not all(dfmgr.dtypes == 'float64'):
                st.error("Data heatmap harus numerik. Harap periksa dataset Anda.")
            else:
                st.subheader("Migrasi Masuk dan Keluar per Kelurahan Kota Bekasi (2022 - 2023)")
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                dfmgr,  
                fmt="d",  
                cmap="coolwarm",  
                linewidths=0.5, 
                linecolor='gray', 
                cbar_kws={"label": "Jumlah Migrasi"}  
            )
            ax_heatmap.set_xlabel("Jenis Migrasi", fontsize=12)
            ax_heatmap.set_ylabel("Kelurahan", fontsize=12)
            st.pyplot(fig_heatmap)

            st.markdown("""
            Grafik diatas menunjukkan **distribusi migrasi masuk dan keluar** di berbagai wilayah kelurahan **Kota Bekasi** selama **2022-2023**. 
            Warna biru hingga merah mencerminkan jumlah migrasi, dengan **merah** menunjukkan **migrasi yang tinggi**. 
            Sebagian besar wilayah memiliki migrasi masuk dan keluar yang seimbang, namun beberapa wilayah, seperti Jakasampurna, menunjukkan migrasi keluar yang sangat tinggi. 
            Perbedaan pola migrasi ini memberikan wawasan penting untuk perencanaan kota, terutama dalam penyediaan fasilitas publik dan analisis ketimpangan antarwilayah.
            """) 
       
    st.text("")  # Menambahkan baris kosong

    col1, col2 = st.columns(2)
    with col1:    
            st.subheader("Proporsi Migrasi Kota Bekasi (2022 - 2023)")
            total_masuk = dfmgr['total_migrasi_masuk'].sum()
            total_keluar = dfmgr['total_migrasi_keluar'].sum()
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            ax_pie.pie([total_masuk, total_keluar], labels=['Migrasi_Masuk', 'Migrasi_Keluar'], autopct='%1.1f%%', startangle=90)
            st.pyplot(fig_pie)
    with col2:
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.markdown("""
            Grafik di samping menunjukkan proporsi antara **migrasi masuk** dan **migrasi keluar** di **Kota Bekasi** periode tahun **2022 - 2023**. 
            Terlihat bahwa migrasi keluar mendominasi dengan persentase sebesar **53.8%**, sedangkan migrasi masuk mencakup **46.2%** dari total migrasi. 
            Data ini menggambarkan bahwa jumlah penduduk yang keluar dari wilayah di Kota Bekasi **lebih besar** dibandingkan dengan jumlah yang masuk. 
            Informasi ini penting untuk **mengidentifikasi pola migrasi** dan **memahami dinamika kependudukan** di wilayah Kota Bekasi.
            """)      

    st.subheader("Cluster Data Migrasi Kota Bekasi (2022 - 2023)")
    st.write(X)
    st.markdown("""
            Berdasarkan hasil analisis cluster terhadap data migrasi **Kota Bekasi** tahun **2022-2023**, terdapat pembagian wilayah ke dalam dua cluster utama, yaitu cluster 0 dengan keterangan **"Peningkatan"** dan cluster 1 dengan keterangan **"Penurunan"**. 
            Wilayah dengan cluster **"Peningkatan"** memiliki angka migrasi keluar yang lebih **tinggi** dibandingkan migrasi masuk, seperti **Bekasijaya** yang memiliki migrasi masuk sebanyak 5.989 orang tetapi migrasi keluar mencapai 7.336 orang. 
            Hal ini mencerminkan adanya kecenderungan **peningkatan** perpindahan penduduk ke luar wilayah.
            
            Sebaliknya, wilayah dengan cluster **"Penurunan"** menunjukkan angka migrasi masuk yang lebih **rendah** dibandingkan migrasi keluar (angka migrasi keluar suatu wilayah lebih rendah dari wilayah lain), 
            misalnya **Kranji** dengan migrasi keluar sebanyak 5.157 orang terlihat **menurun** dari angka migrasi keluar **Bintara** sebanyak 6.112 orang. 
            Fenomena ini mengindikasikan adanya kecenderungan **berkurangnya** arus migrasi penduduk di wilayah tersebut.
            """) 

# Halaman Deskripsi
if menu == "Deskripsi":
    st.header("Deskripsi Aplikasi")
    st.markdown("""
    Aplikasi ini digunakan untuk melakukan analisis clustering pada data migrasi.
    Anda dapat melakukan :
    - Unggah dataset migrasi (.csv).
    - Melakukan preprocessing data (normalisasi).
    - Menentukan jumlah cluster menggunakan Elbow Method.
    - Melihat hasil clustering melalui visualisasi interaktif.
    - Mendownload hasil clustering dalam format CSV.
    """)

# Halaman Unggah Data
if menu == "Unggah Data":
    st.header("Unggah Dataset Migrasi")
    uploaded_file = st.file_uploader("Unggah file dataset (.csv)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset yang diunggah:")
        st.write(df)
        st.session_state['df'] = df

        # Pilih kolom migrasi masuk dan keluar
        st.subheader("Pilih Kolom untuk Migrasi")
        migrasi_masuk_col = st.selectbox(
            "Pilih kolom untuk total migrasi masuk:", df.columns
        )

        migrasi_keluar_col = st.selectbox(
            "Pilih kolom untuk total migrasi keluar:", df.columns
        )

        # Simpan pilihan kolom ke session_state
        st.session_state['migrasi_masuk_col'] = migrasi_masuk_col
        st.session_state['migrasi_keluar_col'] = migrasi_keluar_col

# Halaman Preprocessing
if menu == "Preprocessing":
    st.header("Preprocessing Data")
    
    # Cek apakah dataset sudah diunggah
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df'].copy()  # Copy dataset agar tidak memengaruhi data asli di session_state

        # Ambil nama kolom migrasi masuk dan keluar dari session_state
        migrasi_masuk_col = st.session_state.get('migrasi_masuk_col')
        migrasi_keluar_col = st.session_state.get('migrasi_keluar_col')

        if migrasi_masuk_col and migrasi_keluar_col:
            # Normalisasi data
            scaler = MinMaxScaler()
            X = df[[migrasi_masuk_col, migrasi_keluar_col]]  # Kolom yang dipilih pengguna
            X_normalized = scaler.fit_transform(X)

            # Simpan hasil normalisasi dan scaler ke session_state
            st.session_state['scaler'] = scaler
            st.session_state['X_normalized'] = X_normalized

            # Tampilkan hasil normalisasi
            st.write("Data setelah dinormalisasi:")
            st.write(pd.DataFrame(X_normalized, columns=[migrasi_masuk_col, migrasi_keluar_col]))
        else:
            st.error("Harap pilih kolom migrasi masuk dan keluar terlebih dahulu di menu 'Unggah Data'.")
    else:
        st.warning("Harap unggah dataset terlebih dahulu di menu 'Unggah Data'.")

# Halaman Clustering
if menu == "Clustering":
    st.header("Clustering")
    if st.session_state['X_normalized'] is not None:
        X_normalized = st.session_state['X_normalized']
        migrasi_masuk_col = st.session_state['migrasi_masuk_col']
        migrasi_keluar_col = st.session_state['migrasi_keluar_col']

        # Elbow Method
        st.subheader("Elbow Method")
        k_range = range(1, 11)
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_normalized)
            inertias.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(k_range, inertias, marker='o')
        ax.set_xlabel("Jumlah Cluster (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        # Slider untuk memilih jumlah cluster
        st.subheader("Pilih Jumlah Cluster")
        num_clusters = st.slider("Jumlah Cluster:", 2, 10, 3)

        # Jalankan K-Means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X_normalized)
        df = st.session_state['df']
        df['cluster'] = kmeans.labels_

        # Simpan hasil ke session_state
        st.session_state['df'] = df
        st.session_state['kmeans'] = kmeans
        st.session_state['num_clusters'] = num_clusters

        st.write("Hasil Clustering:")
        st.write(df)
    else:
        st.warning("Harap lakukan preprocessing terlebih dahulu di menu 'Preprocessing'.")
        st.stop()

# Halaman Visualisasi
if menu == "Visualisasi":
    st.header("Visualisasi Clustering")
    
    # Cek apakah data, kmeans, dan scaler tersedia
    if 'df' in st.session_state and 'kmeans' in st.session_state and 'scaler' in st.session_state:
        df = st.session_state['df']
        kmeans = st.session_state['kmeans']
        scaler = st.session_state['scaler']

        # Ambil nama kolom untuk migrasi masuk dan keluar dari session_state
        migrasi_masuk_col = st.session_state.get('migrasi_masuk_col', 'Migrasi Masuk')
        migrasi_keluar_col = st.session_state.get('migrasi_keluar_col', 'Migrasi Keluar')

        # Scatter Plot
        st.subheader("Scatter Plot Clustering")
        fig, ax = plt.subplots(figsize=(10, 5))

        # Scatter plot untuk data cluster
        scatter = ax.scatter(
            df[migrasi_masuk_col],
            df[migrasi_keluar_col],
            c=df['cluster'],
            cmap='rainbow',
            alpha=0.7
        )

        # Centroid (denormalisasi ke skala asli)
        centroids_denorm = scaler.inverse_transform(kmeans.cluster_centers_)

        # Debugging: Menampilkan posisi centroid
        st.write("Centroid Positions (denormalized):")
        st.write(pd.DataFrame(centroids_denorm, columns=[migrasi_masuk_col, migrasi_keluar_col]))

        # Loop untuk menampilkan semua centroid
        for i, centroid in enumerate(centroids_denorm):
            ax.scatter(
                centroid[0],  # Koordinat X denormalisasi
                centroid[1],  # Koordinat Y denormalisasi
                marker='*',      # Bentuk marker
                s=200,           # Ukuran marker
                label=f'Centroid Cluster {i}',
                color=scatter.cmap(i / kmeans.n_clusters)
            )
        
        # Pengaturan sumbu dan legenda
        ax.set_xlabel(migrasi_masuk_col)
        ax.set_ylabel(migrasi_keluar_col)
        ax.set_title("Scatter Plot Clustering")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Data atau model belum tersedia. Silakan lakukan preprocessing dan clustering terlebih dahulu.")


# Halaman Download
if menu == "Download Hasil":
    st.header("Download Hasil Clustering")
    if st.session_state['df'] is not None:
        df = st.session_state['df']

        # Tombol download
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(df)
        st.download_button(
            label="Download Hasil Clustering",
            data=csv,
            file_name="hasil_clustering.csv",
            mime="text/csv",
        )
    else:
        st.warning("Harap lakukan clustering terlebih dahulu di menu 'Clustering'.")
