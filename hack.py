import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import speech_recognition as sr
import pyttsx3
import random

# ------------------ DB Setup ------------------
conn = sqlite3.connect('stress_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS stress_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    stress_level TEXT,
    cause TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# ------------------ Keywords & Advice ------------------
CAUSE_KEYWORDS = {
    "Academic": ["exam", "assignment", "project", "deadline", "grades", "homework", "study"],
    "Social": ["friend", "alone", "lonely", "relationship", "group", "classmates"],
    "Burnout": ["tired", "exhausted", "overwhelmed", "sleep", "burned out", "stress"],
    "Career": ["resume", "internship", "placement", "job", "interview"]
}

ADVICE = {
    "Academic": "Try Pomodoro (25/5). Plan tasks & ask faculty for help.",
    "Social": "Talk to a friend or counselor. Join a campus group.",
    "Burnout": "Take rest, short walks, meditate, or consult health services.",
    "Career": "Reach out to mentors or placement cell for guidance.",
    "General": "Take short breaks, breathe deeply, and relax."
}

# ------------------ Functions ------------------
def analyze_query(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.2:
        stress_level = "High"
    elif polarity < 0:
        stress_level = "Medium"
    else:
        stress_level = "Low"
    cause = "General"
    for key, keywords in CAUSE_KEYWORDS.items():
        if any(word.lower() in text.lower() for word in keywords):
            cause = key
            break
    return {"text": text, "stress_level": stress_level, "cause": cause, "polarity": polarity}

def save_to_db(text, stress_level, cause):
    c.execute('INSERT INTO stress_logs (text, stress_level, cause) VALUES (?,?,?)',
              (text, stress_level, cause))
    conn.commit()

def load_all_records():
    df = pd.read_sql_query('SELECT * FROM stress_logs', conn)
    return df

def display_session_graphs(df_session):
    st.write("### 📊 Session Analysis")
    st.subheader("Stress Level Distribution")
    st.bar_chart(df_session['stress_level'].value_counts())
    
    st.subheader("Cause Distribution")
    st.bar_chart(df_session['cause'].value_counts())
    
    st.subheader("Wordcloud")
    wc = WordCloud(width=600, height=300, background_color="white", colormap='coolwarm').generate(" ".join(df_session['text']))
    fig, ax = plt.subplots(figsize=(8,3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ------------------ Streamlit Layout ------------------
st.set_page_config(page_title="MindTrack Chatbot", layout="wide")
st.markdown("<h1 style='color:#4B0082;'>🧠 MindTrack: AI-Powered Stress Chatbot</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='color:#4B4B4B; font-size:16px'>
Welcome to MindTrack! Choose a feature from the sidebar to analyze messages, detect stress, get advice, and view interactive visualizations.
</p>
""", unsafe_allow_html=True)

# ------------------ Sidebar Feature Selection ------------------
feature = st.sidebar.radio("Choose Feature", [
    "💬 Text Input",
    "🎤 Voice Input",
    "📁 CSV Upload",
    "📥 Simulated Google Classroom",
    "🕵️ MindCheck: Stress Detective"
])

# ------------------ Text Input ------------------
if feature == "💬 Text Input":
    st.markdown("<h2 style='color:#1E90FF; background-color:#F0F0F0; padding:5px; border-radius:5px;'>💬 Chat with MindTrack</h2>", unsafe_allow_html=True)
    user_msg = st.text_input("Type your message here")
    if st.button("Analyze Message", key="text_input"):
        if user_msg.strip():
            result = analyze_query(user_msg)
            save_to_db(result['text'], result['stress_level'], result['cause'])
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"**MindTrack:** Stress: {result['stress_level']} | Cause: {result['cause']}")
            st.info(ADVICE.get(result['cause']))

            session_df = pd.DataFrame([result])
            display_session_graphs(session_df)

# ------------------ Voice Input ------------------
elif feature == "🎤 Voice Input":
    st.markdown("<h2 style='color:#1E90FF; background-color:#F0F0F0; padding:5px; border-radius:5px;'>🎤 Voice Input</h2>", unsafe_allow_html=True)
    if st.button("Click to Record & Analyze", key="voice_input"):
        r = sr.Reco
