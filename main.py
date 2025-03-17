import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Judul aplikasi
st.set_page_config(layout="wide")  # Mengatur tampilan menjadi fullscreen
st.title("Flavor Mapping Viewer")

# Upload file
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

if uploaded_file:
    # Tombol manual untuk refresh data
    if st.button("Refresh Data"):
        st.experimental_rerun()
    
    # Membaca file Excel
    df = pd.read_excel(uploaded_file)
    
    # Pastikan kolom "Dosage (%)" bertipe numerik
    df["Dosage (%)"] = pd.to_numeric(df["Dosage (%)"], errors='coerce')
    
    # Pastikan kolom Character ada dalam dataset
    if "Character" not in df.columns:
        df["Character"] = ""
    
    # Pilihan filter berdasarkan Flavor Group, diurutkan sesuai abjad
    flavor_groups = sorted(df["Flavor Group"].dropna().unique())
    selected_flavor_group = st.selectbox("Pilih Flavor Group", ["All"] + list(flavor_groups))
    
    # Filter data berdasarkan pilihan pengguna
    if selected_flavor_group != "All":
        df = df[df["Flavor Group"] == selected_flavor_group]
    
    # Membuat pivot table
    pivot_table = pd.pivot_table(
        df, 
        values="Dosage (%)", 
        index=["Group", "Name", "Character"], 
        columns=["Flavor Code"], 
        aggfunc="sum",
        fill_value=0
    )
    
    # Konversi semua nilai dalam pivot table menjadi float dan ganti NaN dengan 0
    pivot_table = pivot_table.astype(float).fillna(0)
    
    # Menentukan nilai maksimum dan minimum untuk color scale per kolom (Flavor Code)
    col_mins = pivot_table.min()
    col_maxs = pivot_table.max()
    
    def color_scale_per_column(val, col_min, col_max):
        if isinstance(val, (int, float)):
            if val == 0:
                return "background-color: #D3D3D3; color: black;"  # Blok abu-abu untuk nilai nol
            normalized = (val - col_min) / (col_max - col_min) if col_max > col_min else 0.5
            r = int(200 * (1 - normalized) + 55)  # Lebih smooth gradasi
            g = int(200 * normalized + 55)  # Lebih smooth gradasi
            b = 50
            return f"background-color: rgb({r},{g},{b}); color: black;"
        return ""
    
    def apply_column_wise_styling(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in df.columns:
            styles[col] = df[col].apply(lambda val: color_scale_per_column(val, col_mins[col], col_maxs[col]))
        return styles
    
    styled_pivot = pivot_table.style.apply(apply_column_wise_styling, axis=None).format(precision=4)
    
    df_reset = pivot_table.reset_index()
    
    def render_html_table(df):
        html = "<table border='1' style='border-collapse: collapse; width: 100%; text-align: left;'>"
        html += "<tr style='background-color: #f2f2f2;'>" + "".join([f"<th>{col}</th>" for col in df.columns]) + "</tr>"
        rowspan_count = {}
        rows = []
        
        for i, row in df.iterrows():
            group_val = row["Group"]
            if group_val not in rowspan_count:
                rowspan_count[group_val] = 1
            else:
                rowspan_count[group_val] += 1
            rows.append(row)
        
        for i, row in enumerate(rows):
            group_val = row["Group"]
            html += "<tr>"
            if i == 0 or rows[i - 1]["Group"] != group_val:
                html += f"<td rowspan='{rowspan_count[group_val]}'>{group_val}</td>"
            html += f"<td style='text-align: left;'>{row['Name']}</td>"
            html += f"<td style='color: purple; font-style: italic;'>{row['Character']}</td>"
            
            for col, val in zip(df.columns[3:], row[3:].values):
                if isinstance(val, (int, float)):
                    if val == 0:
                        html += f"<td style='background-color:#D3D3D3; color:black;'>{val:.4f}</td>"
                    else:
                        normalized = (val - col_mins[col]) / (col_maxs[col] - col_mins[col]) if col_maxs[col] > col_mins[col] else 0.5
                        r = int(200 * (1 - normalized) + 55)
                        g = int(200 * normalized + 55)
                        b = 50
                        html += f"<td style='background-color:rgb({r},{g},{b}); color:black;'>{val:.4f}</td>"
                else:
                    html += f"<td>{val}</td>"
            html += "</tr>"
        
        html += "</table>"
        return html
    
    html_table = render_html_table(df_reset)
    
    st.write("### Result Data")
    st.markdown(html_table, unsafe_allow_html=True)
    
    csv = pivot_table.to_csv(float_format="%.4f").encode("utf-8")
    st.download_button(
        "Download hasil sebagai CSV", csv, "pivot_table.csv", "text/csv"
    )

# Layout 3 kolom untuk menampilkan grafik
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### 📊 Grafik Bar Berdasarkan Group") 
        group_data = df.groupby("Group")["Dosage (%)"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))  # Ukuran lebih kecil
        sns.barplot(x="Group", y="Dosage (%)", data=group_data, ax=ax, palette="viridis")
        ax.set_xlabel("Group")
        ax.set_ylabel("Total Dosage (%)")
        ax.set_title("Total Dosage per Group")
        plt.xticks(rotation=90)
        st.pyplot(fig)
    
    with col2:
        st.write("### 📊 Grafik Bar Berdasarkan Flavor Code")
        for flavor_code in df["Flavor Code"].dropna().unique():
            st.write(f"#### {flavor_code}")
            flavor_data = df[df["Flavor Code"] == flavor_code].groupby("Group")["Dosage (%)"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Group", y="Dosage (%)", data=flavor_data, ax=ax, palette="coolwarm")
            ax.set_xlabel("Group")
            ax.set_ylabel("Total Dosage (%)")
            ax.set_title(f"Total Dosage for {flavor_code}")
            plt.xticks(rotation=90)
            st.pyplot(fig)
    
    with col3:
        st.write("### 📊 Grafik Distribusi Dosage (%)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Dosage (%)"], bins=20, kde=True, color="blue", ax=ax)
        ax.set_xlabel("Dosage (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Dosage (%)")
        st.pyplot(fig)

# Membuat dua kolom untuk heatmap dan AI Clustering
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Heatmap Korelasi")
        correlation_matrix = pivot_table.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap of Flavor Codes")
        st.pyplot(fig)
    
    with col2:
        st.write("### Clustering Karakter menggunakan K-Means")
        if not df["Character"].isnull().all():
            le = LabelEncoder()
            df["Character_Encoded"] = le.fit_transform(df["Character"].astype(str))
            
            num_clusters = st.slider("Pilih jumlah cluster:", min_value=2, max_value=10, value=3)
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df["Cluster"] = kmeans.fit_predict(df[["Character_Encoded", "Dosage (%)"]].dropna())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df["Dosage (%)"], y=df["Character"], hue=df["Cluster"], palette="viridis", ax=ax)
            ax.set_xlabel("Dosage (%)")
            ax.set_ylabel("Character")
            ax.set_title("Clustering Distribution of Character using K-Means")
            st.pyplot(fig)
    
    # Analisis clustering
    st.write("### Analisis Clustering K-Means")
    st.markdown(
        """
        **Interpretasi Clustering K-Means:**
        - AI mengelompokkan karakter berdasarkan **kesamaan pola pemakaian Flavor Code** dan **dosage (%)**.
        - Setiap warna dalam scatter plot menunjukkan kelompok berbeda yang memiliki karakteristik yang mirip.
        - Cluster ini dapat membantu dalam memahami pola penggunaan Flavor Code dalam berbagai karakter.
        
        **Manfaat Clustering ini:**
        - Mengidentifikasi **karakter yang sering digunakan bersama**.
        - Membantu formulasi dengan menemukan pola tersembunyi dari distribusi karakter.
        - Memberikan insight apakah ada karakter yang memiliki **penggunaan yang sangat unik atau mirip dengan lainnya**.
        """
    )
    

    # Footer about program
    st.markdown("---")
    st.markdown(
        """
        ### About Program
        - **Program Analisis Data** untuk memvisualisasikan hubungan antar Flavor Code dan pola penggunaannya.
        - Dilengkapi dengan **Heatmap Korelasi** untuk melihat keterkaitan antar Flavor Code.
        - Menggunakan **K-Means Clustering AI** untuk menganalisis persebaran karakter dalam dataset.
        - Dibuat dengan ❤️ oleh **Mr-XP**.
        """
    )
