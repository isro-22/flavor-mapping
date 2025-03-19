import streamlit as st
import pandas as pd
import sqlite3
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

class FlavorMappingViewer:
    def __init__(self):
        st.set_page_config(layout="wide")
        st.title("Flavor Mapping Viewer")
        self.uploaded_file = st.file_uploader("Upload Excel or SQL file", type=["xlsx", "sql"])
        if self.uploaded_file:
            if st.button("Refresh Data"):
                st.rerun()
            self.process_file()

    def process_file(self):
        if self.uploaded_file.name.endswith('.xlsx'):
            self.df = pd.read_excel(self.uploaded_file)
        elif self.uploaded_file.name.endswith('.sql'):
            self.process_sql_file()
        
        self.clean_and_prepare_data()
        self.filter_data()
        self.create_pivot_table()
        self.display_pivot_table()
        self.display_charts()
        self.display_heatmap_and_clustering() 
        self.footer_web()

    def process_sql_file(self):
        sql_file = self.uploaded_file.read().decode('utf-8')
        sql_file = re.sub(r'\bnan\b', 'NULL', sql_file)
        try:
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            cursor.executescript(sql_file)
            self.df = pd.read_sql_query("SELECT * FROM flavor_data", conn)
            conn.close()
        except sqlite3.OperationalError as e:
            st.error(f"SQL Error: {e}")
            st.stop()

    def clean_and_prepare_data(self):
        # Ensure Dosage_() is numeric
        self.df["Dosage_()"] = pd.to_numeric(self.df["Dosage_()"], errors='coerce')
        
        # Remove rows with NaN in Dosage_()
        self.df.dropna(subset=["Dosage_()"], inplace=True)
        
        # Ensure other columns do not contain invalid values
        self.df = self.df[self.df["Flavor_Code"].notna()]
        self.df = self.df[self.df["Functional_Group"].notna()]
        
        # Fill empty values in the Character column with an empty string
        if "Character" not in self.df.columns:
            self.df["Character"] = ""
        self.df["Character"].fillna("", inplace=True)

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
        self.pivot_table.sort_values(by=["Functional_Group_Empty", "Functional_Group"], ascending=[True, True], inplace=True)
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
                    normalized = (val - self.col_mins[col]) / (self.col_maxs[col] - self.col_mins[col]) if self.col_maxs[col] > self.col_mins[col] else 0.5
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
            flavor_data = self.df[self.df["Flavor_Code"] == flavor_code].groupby("Functional_Group")["Dosage_()"].sum().reset_index()
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
            # Ensure only numeric columns are used
            numeric_columns = self.pivot_table.select_dtypes(include=[np.number])
            correlation_matrix = numeric_columns.corr()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Correlation Heatmap of Flavor Codes")
            st.pyplot(fig)
        
        with col2:
            st.write("### Character Clustering using K-Means")
            if not self.df["Character"].isnull().all():
                # Encode the Character column into numeric
                le = LabelEncoder()
                self.df["Character_Encoded"] = le.fit_transform(self.df["Character"].astype(str))
                
                # Use numeric columns for clustering
                num_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                self.df["Cluster"] = kmeans.fit_predict(self.df[["Character_Encoded", "Dosage_()"]].dropna())
                
                # Display scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=self.df["Dosage_()"], y=self.df["Character"], hue=self.df["Cluster"], palette="viridis", ax=ax)
                ax.set_xlabel("Dosage (%)")
                ax.set_ylabel("Character")
                ax.set_title("Clustering Distribution of Character using K-Means")
                st.pyplot(fig)

        # Add heatmap and clustering analysis interpretation
        st.markdown("---")  # Divider visual
        col1, col2 = st.columns(2)

        with col1:
            # Heatmap Interpretation in a card-like layout
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

        # Add a visual divider between the two columns
        st.divider()

        with col2:
            # Clustering Analysis in a card-like layout
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

    def footer_web(self):
        # Footer About the Program
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

if __name__ == "__main__":
    viewer = FlavorMappingViewer()