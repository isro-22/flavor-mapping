import streamlit as st
import pandas as pd
import sqlite3
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Set page configuration at the very beginning
st.set_page_config(layout="wide")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")

# 🔹 Gunakan expander agar sidebar bisa di-hide/unhide
with st.sidebar.expander("⚙️ Menu", expanded=True):
    page = st.radio(
        "📌 Go to",
        ["Flavor Mapping", "Flavor Compare"],
        index=0
    )

# 🔹 Pisahkan menu dengan garis pemisah
st.sidebar.divider()

# 🔹 Tambahkan informasi atau credit di bagian bawah sidebar
st.sidebar.markdown("📊 **Flavor Data Analysis Tool**")
st.sidebar.markdown(
    "<p style='font-size:10px; color:gray;'>© 2025. Flavor-House. All rights reserved.</p>",
    unsafe_allow_html=True
)


# Cache data processing functions
@st.cache_data  # Use st.cache_data for caching data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.sql'):
        df = process_sql_file(uploaded_file)
    return df


@st.cache_data  # Use st.cache_data for caching data
def process_sql_file(uploaded_file):
    sql_file = uploaded_file.read().decode('utf-8')
    sql_file = re.sub(r'\bnan\b', 'NULL', sql_file)

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.executescript(sql_file)

    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    table_names = [t[0] for t in tables]

    if "flavor_data" in table_names:
        df = pd.read_sql_query("SELECT * FROM flavor_data", conn)
    else:
        st.error("Table 'flavor_data' not found in SQL file!")
        return None

    conn.close()
    return df


def standardize_column_names(df):
    rename_map = {
        "Dosage (%)": "Dosage_()",
        "Flavor Code": "Flavor_Code",
        'Functional Group': "Functional_Group",
        'Flavor Group': "Flavor_Group",
        'Character Group': "Character_Group"
    }
    return df.rename(columns=rename_map)


def clean_and_prepare_data(df):
    df["Dosage_()"] = pd.to_numeric(df["Dosage_()"], errors='coerce')
    df.dropna(subset=["Dosage_()"], inplace=True)
    df = df[df["Flavor_Code"].notna()]
    df["Functional_Group"].fillna("", inplace=True)
    if "Character" not in df.columns:
        df["Character"] = ""
    df["Character"].fillna("", inplace=True)
    return df


# Page 1: Flavor Mapping Viewer
class FlavorMappingViewer:
    def __init__(self):
        st.title("Flavor Mapping Viewer")

        if "shared_data" in st.session_state and "file_name" in st.session_state:
            st.success(f"Using previously uploaded file: {st.session_state['file_name']}")
            self.df = st.session_state["shared_data"]
            self.display_data()
        else:
            self.uploaded_file = st.file_uploader("Upload Excel or SQL file", type=["xlsx", "sql"])
            if self.uploaded_file:
                st.session_state["file_name"] = self.uploaded_file.name
                if st.button("Load Data"):
                    self.df = load_data(self.uploaded_file)
                    self.df = standardize_column_names(self.df)
                    self.df = clean_and_prepare_data(self.df)
                    st.session_state["shared_data"] = self.df
                    self.display_data()

    def display_data(self):
        self.filter_data()
        self.create_pivot_table()
        self.display_pivot_table()
        self.display_charts()
        self.display_heatmap_and_clustering()
        display_footer()

    def filter_data(self):
        flavor_groups = sorted(self.df["Flavor_Group"].dropna().unique())
        self.selected_flavor_group = st.selectbox("Select Flavor Group", ["All"] + list(flavor_groups))
        if self.selected_flavor_group != "All":
            self.df = self.df[self.df["Flavor_Group"] == self.selected_flavor_group]

    def create_pivot_table(self):
        self.pivot_table = pd.pivot_table(
            self.df,
            values="Dosage_()",
            index=["Functional_Group", "Name", "Character"],
            columns=["Flavor_Code"],
            aggfunc="sum",
            fill_value=0
        )
        self.pivot_table = self.pivot_table.astype(float).fillna(0)
        self.pivot_table.reset_index(inplace=True)
        self.pivot_table.iloc[:, 3:] = self.pivot_table.iloc[:, 3:].applymap(lambda x: round(x, 5))
        self.pivot_table["Functional_Group_Empty"] = self.pivot_table["Functional_Group"] == ""
        self.pivot_table.sort_values(by=["Functional_Group_Empty", "Functional_Group"], ascending=[True, True],
                                     inplace=True)
        self.pivot_table.drop(columns=["Functional_Group_Empty"], inplace=True)
        self.col_mins = self.pivot_table.iloc[:, 3:].min()
        self.col_maxs = self.pivot_table.iloc[:, 3:].max()

    def render_html_table(self, df):
        html = "<table border='1' style='border-collapse: collapse; width: 100%; text-align: left;'>"
        html += "<tr style='background-color: #f2f2f2;'>" + "".join([f"<th>{col}</th>" for col in df.columns]) + "</tr>"
        rowspan_count = df.groupby("Functional_Group").size().to_dict()
        prev_group = None

        for i, row in df.iterrows():
            group_val = row["Functional_Group"]
            html += "<tr>"
            if group_val != prev_group:
                html += f"<td rowspan='{rowspan_count[group_val]}'>{group_val}</td>"
            html += f"<td>{row['Name']}</td>"
            html += f"<td style='color: purple; font-style: italic;'>{row['Character']}</td>"

            for col in df.columns[3:]:
                val = row[col]
                if val == 0:
                    bg_color = "#D3D3D3"  # Grey for zero values
                else:
                    normalized = (val - self.col_mins[col]) / (self.col_maxs[col] - self.col_mins[col]) if \
                    self.col_maxs[col] > self.col_mins[col] else 0.5
                    r = int(255 * (1 - normalized))
                    g = int(255 * normalized)
                    b = 50
                    bg_color = f"rgb({r},{g},{b})"
                html += f"<td style='background-color:{bg_color}; color:black;'>{val:.5f}</td>"
            html += "</tr>"
            prev_group = group_val

        html += "</table>"
        return html

    def display_pivot_table(self):
        html_table = self.render_html_table(self.pivot_table)
        st.write("### Data Results")
        st.markdown(html_table, unsafe_allow_html=True)
        st.markdown("---")

    def display_charts(self):
        col1, col2, col3 = st.columns(3)
        st.markdown("---")
        with col1:
            self.display_functional_group_chart()
        with col2:
            self.display_flavor_code_charts()
        with col3:
            self.display_dosage_distribution_chart()

    def display_functional_group_chart(self):
        st.write("### 📊 Bar Chart by Functional Group")
        group_data = self.df.groupby("Functional_Group")["Dosage_()"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Functional_Group", y="Dosage_()", data=group_data, ax=ax, palette="viridis")
        ax.set_xlabel("Functional Group")
        ax.set_ylabel("Total Dosage (%)")
        ax.set_title("Total Dosage per Functional Group")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    def display_flavor_code_charts(self):
        st.write("### 📊 Bar Chart by Flavor Code")
        unique_flavors = self.df["Flavor_Code"].dropna().unique()
        for flavor_code in unique_flavors:
            flavor_data = self.df[self.df["Flavor_Code"] == flavor_code].groupby("Functional_Group")[
                "Dosage_()"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Functional_Group", y="Dosage_()", data=flavor_data, ax=ax, palette="coolwarm")
            ax.set_xlabel("Functional Group")
            ax.set_ylabel("Total Dosage (%)")
            ax.set_title(f"Total Dosage for {flavor_code}")
            plt.xticks(rotation=90)
            st.pyplot(fig)

    def display_dosage_distribution_chart(self):
        st.write("### 📊 Dosage Distribution Chart")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df["Dosage_()"], bins=20, kde=True, color="blue", ax=ax)
        ax.set_xlabel("Dosage (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Dosage (%)")
        st.pyplot(fig)

    def display_heatmap_and_clustering(self):
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Correlation Heatmap")
            numeric_columns = self.pivot_table.select_dtypes(include=[np.number])
            correlation_matrix = numeric_columns.corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Correlation Heatmap of Flavor Codes")
            st.pyplot(fig)

        with col2:
            st.write("### Character Clustering using K-Means")
            if not self.df["Character"].isnull().all():
                le = LabelEncoder()
                self.df["Character_Encoded"] = le.fit_transform(self.df["Character"].astype(str))

                num_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                self.df["Cluster"] = kmeans.fit_predict(self.df[["Character_Encoded", "Dosage_()"]].dropna())

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=self.df["Dosage_()"], y=self.df["Character"], hue=self.df["Cluster"],
                                palette="viridis", ax=ax)
                ax.set_xlabel("Dosage (%)")
                ax.set_ylabel("Character")
                ax.set_title("Clustering Distribution of Character using K-Means")
                st.pyplot(fig)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div style="
                    padding: 10px;
                    border-radius: 10px;
                    border: 1px solid #e0e0e0;
                    background-color: #f9f9f9;
                    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                ">
                <h4 style="color: #2e86c1;">Heatmap Interpretation</h4>
                <p>
                <strong>Interpretation of Correlation Heatmap:</strong><br>
                - The heatmap shows the <strong>correlation between Flavor Codes</strong> based on their usage patterns.<br>
                - <strong>Values range from -1 to 1:</strong><br>
                &nbsp;&nbsp;- <strong>1</strong>: Perfect positive correlation (Flavor Codes are used together frequently).<br>
                &nbsp;&nbsp;- <strong>-1</strong>: Perfect negative correlation (Flavor Codes are rarely used together).<br>
                &nbsp;&nbsp;- <strong>0</strong>: No correlation (Flavor Codes have no relationship in usage).<br>
                - <strong>Warm colors (red/orange)</strong>: Indicate strong positive correlations.<br>
                - <strong>Cool colors (blue)</strong>: Indicate strong negative correlations.<br>
                - <strong>Insights:</strong><br>
                &nbsp;&nbsp;- Identify <strong>Flavor Codes that are often used together</strong> (high positive correlation).<br>
                &nbsp;&nbsp;- Detect <strong>Flavor Codes that are mutually exclusive</strong> (high negative correlation).<br>
                &nbsp;&nbsp;- Understand <strong>which Flavor Codes have no relationship</strong> (values close to 0).<br>
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        with col2:
            st.markdown(
                """
                <div style="
                    padding: 10px;
                    border-radius: 10px;
                    border: 1px solid #e0e0e0;
                    background-color: #f9f9f9;
                    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                ">
                <h4 style="color: #2e86c1;">K-Means Clustering Analysis</h4>
                <p>
                <strong>K-Means Clustering Interpretation:</strong><br>
                - AI groups characters based on <strong>similar usage patterns of Flavor Code</strong> and <strong>dosage (%)</strong>.<br>
                - Each color in the scatter plot represents a different group with similar characteristics.<br>
                - These clusters can help in understanding the usage patterns of Flavor Code across different characters.<br>

                <strong>Benefits of Clustering:</strong><br>
                - Identify <strong>characters that are often used together</strong>.<br>
                - Assist in formulation by uncovering hidden patterns in character distribution.<br>
                - Provide insights into whether there are characters with <strong>unique or similar usage patterns</strong>.<br>
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# Page 2: Data Reference Pivot Table
class DataReference:
    def __init__(self):
        st.title("Data Reference - Flavor Comparison")
        self.load_data()

    def load_data(self):
        if "shared_data" in st.session_state:
            self.df = st.session_state["shared_data"]
            st.write("Using uploaded data from Page 1")
            self.create_filters()
        else:
            st.warning("No data available. Please upload data in Page 1 first.")

    def create_filters(self):
        st.write("### Select Flavor Groups and Codes for Comparison")
        col1, col2, col3 = st.columns(3)

        flavor_groups = sorted(self.df["Flavor_Group"].dropna().unique())
        selected_groups = []
        selected_codes = []

        for i, col in enumerate([col1, col2, col3]):
            with col:
                selected_group = st.selectbox(f"Flavor Group {i + 1}", [None] + list(flavor_groups),
                                              key=f"flavor_group_{i}")
                selected_groups.append(selected_group)

                if selected_group:
                    filtered_df = self.df[self.df["Flavor_Group"] == selected_group]
                    flavor_codes = sorted(filtered_df["Flavor_Code"].dropna().unique())
                    selected_code = st.selectbox(f"Flavor Code {i + 1}", [None] + list(flavor_codes),
                                                 key=f"flavor_code_{i}")
                    selected_codes.append(selected_code)
                else:
                    selected_codes.append(None)

        filtered_df = self.df[self.df["Flavor_Group"].isin([g for g in selected_groups if g])]
        filtered_df = filtered_df[filtered_df["Flavor_Code"].isin([c for c in selected_codes if c])]

        self.create_pivot_table(filtered_df)

    def create_pivot_table(self, df):
        if df.empty:
            st.warning("No data available for the selected filters.")
            return

        pivot_table = pd.pivot_table(
            df,
            values="Dosage_()",
            index=["Name", "Character"],
            columns=["Flavor_Code"],
            aggfunc="sum",
            fill_value=0
        )

        pivot_table.reset_index(inplace=True)
        pivot_table.index = pivot_table.index + 1  # Start numbering from 1

        def highlight_zero(val):
            if val == 0:
                return "color: red;"
            return "color: black;"

        numeric_columns = pivot_table.columns[2:]  # Semua kolom setelah "Name" & "Character"
        styled_df = pivot_table.style.applymap(highlight_zero, subset=numeric_columns)

        styled_df = styled_df.applymap(lambda x: "font-style: italic; color: purple;", subset=["Character"])

        st.write("### Table Comparison")
        st.dataframe(styled_df)

        self.display_bar_charts(df)
        self.display_sunburst_charts(df)
        display_footer()

    def display_bar_charts(self, df):
        st.write("### Dosage Distribution by Flavor Code and Functional Group")

        top_20 = df.sort_values(by="Dosage_()", ascending=False).head(20)

        col1, col2 = st.columns(2)

        with col1:
            st.write("#### By Flavor Code")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x="Dosage_()",
                y="Name",
                hue="Flavor_Code",
                data=top_20,
                palette="tab10",
                ax=ax
            )
            ax.set_xlabel("Dosage (%)")
            ax.set_ylabel("Name")
            ax.set_title("Top 20 Dosage by Flavor Code")
            st.pyplot(fig)

        with col2:
            st.write("#### By Functional Group")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(
                x="Dosage_()",
                y="Name",
                hue="Functional_Group",
                data=top_20,
                palette="Set2",
                ax=ax
            )
            ax.set_xlabel("Dosage (%)")
            ax.set_ylabel("Name")
            ax.set_title("Top 20 Dosage by Functional Group")
            st.pyplot(fig)

    def display_sunburst_charts(self, df):
        st.write("### Flavor Whell for Each Flavor Code")

        unique_flavor_codes = df["Flavor_Code"].unique()
        cols = st.columns(3)

        for i, flavor_code in enumerate(unique_flavor_codes):
            with cols[i % 3]:
                st.write(f"#### {flavor_code}")

                df_flavor = df[df["Flavor_Code"] == flavor_code].copy()

                df_flavor["Group1"] = df_flavor["Character_Group"]

                df_flavor["Group2"] = df_flavor["Character"].str.split(",")

                df_flavor = df_flavor.explode("Group2")

                df_flavor = df_flavor[df_flavor["Group2"].notna() & (df_flavor["Group2"].str.strip() != "")]

                fig = px.sunburst(
                    df_flavor,
                    path=["Group1", "Group2"],
                    values="Dosage_()",
                    color="Group1",
                    color_discrete_sequence=px.colors.sequential.Plasma
                )

                st.plotly_chart(fig)

# Footer Web
def display_footer():
    st.markdown(
        """
        <div style="
            padding: 20px;
            background-color: #2e86c1;
            color: white;
            border-radius: 10px;
            margin-top: 40px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        ">
        <h3 style="text-align: center; color: white;">About the Program</h3>
        <p style="text-align: center;">
            This <strong>Data Analysis Program</strong> is designed to visualize the relationships between Flavor Codes and their usage patterns.<br>
            Equipped with a <strong>Correlation Heatmap</strong> to see the relationships between Flavor Codes.<br>
            Uses <strong>K-Means Clustering AI</strong> to analyze the distribution of characters in the dataset.<br>
            Made with ❤️ by <strong>Mr-XP</strong>.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if page == "Flavor Mapping":
    viewer = FlavorMappingViewer()
elif page == "Flavor Compare":
    DataReference()