import streamlit as st
import os
import hashlib
import pandas as pd
import numpy as np
import datetime
import requests
import speech_recognition as sr
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import cohere
import google.auth
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

# ----------------------- API & Credentials Configuration -----------------------
API_CONFIG = {
    'cohere_api_key': st.secrets["COHERE_API_KEY"],
    'google_calendar_scopes': st.secrets["CALENDAR_API_KEY"],
    'google_credentials_file': st.secrets["CREDENTIALS_API"],
    'URL': st.secrets["URL"]
}

# Initialize Cohere API for the chatbot
cohere_client = cohere.Client(API_CONFIG['cohere_api_key'])

# Set up Google Calendar API
creds = None

# ----------------------- Section 1: User Authentication -----------------------
def authenticate_user(username, password):
    return username == "admin" and password == "password123"

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

st.sidebar.title("Login")
if not st.session_state.authenticated:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button and authenticate_user(username, password):
        st.session_state.authenticated = True
        st.sidebar.success("Login Successful!")
else:
    st.sidebar.success("You are logged in!")

    # ----------------------- Section 2: Document Integrity Check (Blockchain) -----------------------
    st.title("Document Integrity Check")
    uploaded_file = st.file_uploader("Upload a Legal Document", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        st.write(f"Document Hash (SHA-256): `{file_hash}`")

    # ----------------------- Section 3: Legal Research Assistant -----------------------
    st.title("Legal Research Assistant")
    query = st.text_input("Enter a Legal Query:")
    
    if st.button("Search Relevant Case Laws"):
        response = requests.get(f"{API_CONFIG['URL']}{query.replace(' ', '+')}")
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all("div", class_="case-law-summary")
        
        st.write("### Top Relevant Case Laws:")
        for i, result in enumerate(results[:5], start=1):
            st.write(f"**{i}.** {result.text.strip()}")

    # ----------------------- Section 4: Case Outcome Prediction -----------------------
    st.title("Case Outcome Prediction & Analytics")

    sample_data = pd.DataFrame({
        "Case Duration (Days)": np.random.randint(30, 730, 100),
        "Risk Score": np.random.uniform(0.1, 1.0, 100)
    })

    st.write("### Case Data:")
    st.write(sample_data)

    X = sample_data["Risk Score"].values.reshape(-1, 1)
    y = sample_data["Case Duration (Days)"].values
    model = LinearRegression()
    model.fit(X, y)

    risk_score = st.slider("Enter Risk Score for Prediction:", 0.1, 1.0, 0.5)
    predicted_duration = model.predict([[risk_score]])[0]
    st.write(f"Predicted Case Duration: **{int(predicted_duration)} days**")

    fig, ax = plt.subplots()
    ax.scatter(sample_data["Risk Score"], sample_data["Case Duration (Days)"], label="Actual Data")
    ax.plot(sample_data["Risk Score"], model.predict(X), color='red', label="Prediction Line")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Case Duration (Days)")
    ax.legend()
    st.pyplot(fig)

    # ----------------------- Section 5: Court Hearing Scheduler -----------------------
    st.title("Court Hearing Scheduler")
    hearing_date = st.date_input("Select Hearing Date:", datetime.date.today())
    st.write(f"Scheduled Hearing Date: **{hearing_date.strftime('%B %d, %Y')}**")

    # ----------------------- Section 6: Voice Command Integration -----------------------
    st.title("Voice Command Case Search")

    if st.button("Start Listening"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening for a case name or query...")
            try:
                audio = recognizer.listen(source, timeout=5)
                voice_query = recognizer.recognize_google(audio)
                st.write(f"You said: **{voice_query}**")
            except sr.UnknownValueError:
                st.write("Sorry, I didn't catch that.")
            except sr.RequestError:
                st.write("Service unavailable.")

    # ----------------------- Section 7: Cohere Chatbot -----------------------
    st.title("Legal Chatbot")
    user_input = st.text_input("Ask your legal question:")
    
    if st.button("Ask Chatbot"):
        if user_input:
            response = cohere_client.generate(
                model="xlarge",
                prompt=f"Legal expert assistant: {user_input}",
                max_tokens=100,
                temperature=0.5
            )
            st.write(f"**Bot Response:** {response.generations[0].text.strip()}")
        else:
            st.write("Please type a question.")

    # ----------------------- Section 8: Google Calendar API Integration -----------------------
    st.title("Add Hearing to Google Calendar")
    
    if st.button("Authenticate and Connect Google Calendar"):
        flow = InstalledAppFlow.from_client_secrets_file(
            API_CONFIG['google_credentials_file'], API_CONFIG['google_calendar_scopes'])
        creds = flow.run_local_server(port=0)
        
        service = build('calendar', 'v3', credentials=creds)
        
        event = {
            'summary': 'Court Hearing',
            'location': '123 Legal Ave, Courtroom 7',
            'description': 'Scheduled court hearing for case review.',
            'start': {
                'dateTime': hearing_date.strftime('%Y-%m-%dT%H:%M:%S'),
                'timeZone': 'America/New_York',
            },
            'end': {
                'dateTime': (hearing_date + datetime.timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%S'),
                'timeZone': 'America/New_York',
            },
        }
        
        event = service.events().insert(calendarId='primary', body=event).execute()
        st.write(f"Event created: {event.get('htmlLink')}")

else:
    st.sidebar.error("Authentication Failed. Please try again.")
