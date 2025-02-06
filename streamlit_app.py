import streamlit as st
import pandas as pd
import datetime
import re
import requests
from bs4 import BeautifulSoup
import sqlite3
from transformers import pipeline
import matplotlib.pyplot as plt

# Initialize SQLite Database
conn = sqlite3.connect('case_data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS cases (case_id TEXT, document_type TEXT, status TEXT, last_updated TEXT)''')
conn.commit()

# Hugging Face NER model
ner = pipeline("ner", model="dslim/bert-base-NER")

# Streamlit UI
st.title("⚖️ AI-Powered Legal Case Management System")
st.sidebar.header("Features")
option = st.sidebar.selectbox("Select a Task:", ["Streamline Case Management", "Automate Document Processing", "Enhance Decision-Making", "Data Insights & Visualization"])

# Function to fetch case data from the internet
def search_online_case(case_id):
    st.write(f"🔍 Searching for case {case_id} online...")
    url = f"https://mock-legal-website.com/cases/{case_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            case_type = soup.find("span", class_="case-type").text
            status = soup.find("span", class_="case-status").text
            last_updated = soup.find("span", class_="last-updated").text
            return {"Case ID": case_id, "Document Type": case_type, "Status": status, "Last Updated": last_updated}
        else:
            st.error("❌ Case not found online.")
            return None
    except Exception as e:
        st.error(f"⚠️ Error fetching case data: {e}")
        return None

# Document processing with NER and regex
def process_document():
    st.subheader("📄 Upload a Document for Processing")
    uploaded_file = st.file_uploader("Upload a legal document (txt/pdf)", type=["txt", "pdf"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.text_area("Document Content:", text, height=300)

        # Named Entity Recognition (NER) using Hugging Face
        st.write("**Named Entity Recognition (NER):**")
        ner_results = ner(text)
        for entity in ner_results:
            st.write(f"{entity['word']} ({entity['entity']})")

        # Extract Case ID and Dates using regex
        case_ids = re.findall(r"(Case\sID:\s\w+)", text)
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)

        st.write("**Extracted Case Details:**")
        if case_ids:
            st.write("Case IDs:", case_ids)
        if dates:
            st.write("Dates:", dates)
        if not case_ids and not dates:
            st.write("No recognizable patterns found.")

# Function to manage case workflows
def manage_cases():
    st.subheader("🗂️ Case Management Dashboard")
    c.execute("SELECT * FROM cases")
    data = c.fetchall()
    df = pd.DataFrame(data, columns=["Case ID", "Document Type", "Status", "Last Updated"])
    st.dataframe(df)

    st.subheader("🔎 Search or Add a Case")
    case_id = st.text_input("Enter Case ID to search:")
    if st.button("Search Case"):
        case_data = df[df["Case ID"] == case_id]
        if not case_data.empty:
            st.write(f"✅ Case {case_id} found:")
            st.dataframe(case_data)
        else:
            online_case = search_online_case(case_id)
            if online_case:
                st.write("🌐 Case found online:")
                st.write(online_case)
                c.execute("INSERT INTO cases VALUES (?, ?, ?, ?)", tuple(online_case.values()))
                conn.commit()
            else:
                st.error("❌ Case not found.")

# Decision-making with uploaded case data
def enhance_decision_making():
    st.subheader("📊 Decision-Making Assistance")
    uploaded_file = st.file_uploader("Upload case data (CSV format)", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())
        st.write("### 📈 Data Analysis:")
        st.write(f"Total Cases: {len(data)}")
        st.write(f"Closed Cases: {len(data[data['Status'] == 'Closed'])}")
        st.write(f"Pending Cases: {len(data[data['Status'] == 'Pending'])}")

# Data insights and visualization
def visualize_data():
    st.subheader("📊 Data Insights & Visualization")
    c.execute("SELECT * FROM cases")
    data = c.fetchall()
    df = pd.DataFrame(data, columns=["Case ID", "Document Type", "Status", "Last Updated"])
    
    status_counts = df["Status"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(status_counts.index, status_counts.values, color=["green", "orange", "red"])
    ax.set_title("Case Status Distribution")
    ax.set_xlabel("Status")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Main workflow
if option == "Streamline Case Management":
    manage_cases()
elif option == "Automate Document Processing":
    process_document()
elif option == "Enhance Decision-Making":
    enhance_decision_making()
elif option == "Data Insights & Visualization":
    visualize_data()
