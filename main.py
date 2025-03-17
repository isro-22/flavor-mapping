import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Application title
st.set_page_config(layout="wide")  # Set the display to fullscreen
st.title("Flavor Mapping Viewer")

# File upload
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    # Manual button to refresh data
    if st.button("Refresh Data"):
        st.experimental_rerun()
    
    # Read the Excel file
    df = pd.read_excel(uploaded_file)
    
    # Ensure the "Dosage (%)" column is numeric
    df["Dosage (%)"] = pd.to_numeric(df["Dosage (%)"], errors='coerce')
    
    # Ensure the "Character" column exists in the dataset
    if "Character" not in df.columns:
        df["Character"] = ""
    
    # Filter selection based on Flavor Group, sorted alphabetically
    flavor_groups = sorted(df["Flavor Group"].dropna().unique())
    selected_flavor_group = st.selectbox("Select Flavor Group", ["All"] + list(flavor_groups))
    
    # Filter data based on user selection
    if selected_flavor_group != "All":
        df = df[df["Flavor Group"] == selected_flavor_group]
    
    # Create pivot table
    pivot_table = pd.pivot_table(
        df, 
        values="Dosage (%)", 
        index=["Group", "Name", "Character"], 
        columns=["Flavor Code"], 
        aggfunc="sum",
        fill_value=0
    )
    
    # Convert all values in the pivot table to float and replace NaN with 0
    pivot_table = pivot_table.astype(float).fillna(0)
    
    # Determine the maximum and minimum values for color scale per column (Flavor Code)
    col_mins = pivot_table.min()
    col_maxs = pivot_table.max()
    
    def color_scale_per_column(val, col_min, col_max):
        if isinstance(val, (int, float)):
            if val == 0:
                return "background-color: #D3D3D3; color: black;"  # Gray block for zero values
            normalized = (val - col_min) / (col_max - col_min) if col_max > col_min else 0.5
            r = int(200 * (1 - normalized) + 55)  # Smoother gradient
            g = int(200 * normalized + 55)  # Smoother gradient
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
        "Download result as CSV", csv, "pivot_table.csv", "text/csv"
    )

# Layout with 3 columns to display charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### 📊 Bar Chart by Group") 
        group_data = df.groupby("Group")["Dosage (%)"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller size
        sns.barplot(x="Group", y="Dosage (%)", data=group_data, ax=ax, palette="viridis")
        ax.set_xlabel("Group")
        ax.set_ylabel("Total Dosage (%)")
        ax.set_title("Total Dosage per Group")
        plt.xticks(rotation=90)
        st.pyplot(fig)
    
    with col2:
        st.write("### 📊 Bar Chart by Flavor Code")
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
        st.write("### 📊 Dosage (%) Distribution Chart")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Dosage (%)"], bins=20, kde=True, color="blue", ax=ax)
        ax.set_xlabel("Dosage (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Dosage (%)")
        st.pyplot(fig)




# Create two columns for heatmap and AI Clustering
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Correlation Heatmap")
        correlation_matrix = pivot_table.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap of Flavor Codes")
        st.pyplot(fig)
   
    with col2:
        st.write("### Character Clustering using K-Means")
        if not df["Character"].isnull().all():
            le = LabelEncoder()
            df["Character_Encoded"] = le.fit_transform(df["Character"].astype(str))
            
            num_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df["Cluster"] = kmeans.fit_predict(df[["Character_Encoded", "Dosage (%)"]].dropna())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df["Dosage (%)"], y=df["Character"], hue=df["Cluster"], palette="viridis", ax=ax)
            ax.set_xlabel("Dosage (%)")
            ax.set_ylabel("Character")
            ax.set_title("Clustering Distribution of Character using K-Means")
            st.pyplot(fig)
      
    # Create two columns for Heatmap Interpretation and Clustering Analysis
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
