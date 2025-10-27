
import streamlit as st
import pandas as pd
import sqlite3
import random
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import speech_recognition as sr
import pyttsx3


st.set_page_config(page_title="MindTrack", layout="wide")
st.title("MindTrack: AI-Powered Academic Stress Insight System")
st.markdown("**Detects stress levels and causes from text or voice, provides support, and visualizes trends.**")


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
    c.execute('INSERT INTO stress_logs (text, stress_level, cause) VALUES (?,?,?)', (text, stress_level, cause))
    conn.commit()

def load_all_records():
    df = pd.read_sql_query('SELECT * FROM stress_logs', conn)
    return df


st.subheader("1️⃣ Analyze Student Messages or Searches")


txt = st.text_area("Paste multiple messages/searches (one per line)", height=150)
if st.button("Analyze Pasted Messages"):
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    results = [analyze_query(l) for l in lines]
    for r in results:
        save_to_db(r['text'], r['stress_level'], r['cause'])
    df_res = pd.DataFrame(results)
    st.dataframe(df_res[['text','stress_level','cause']])
    st.download_button("Download Results CSV", df_res.to_csv(index=False), "results.csv", "text/csv")


st.subheader("Or Upload a CSV (auto-detects text column)")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df_in = pd.read_csv(uploaded)
    st.write("✅ CSV Loaded Successfully!")
    st.write("Columns found:", list(df_in.columns))

    
    text_col = None
    for col in df_in.columns:
        if "query" in col.lower() or "message" in col.lower() or "text" in col.lower() or "response" in col.lower():
            text_col = col
            break

    if not text_col:
        text_col = st.selectbox("Select the column containing messages/text:", df_in.columns)

    try:
        texts = df_in[text_col].astype(str).tolist()
        results = [analyze_query(t) for t in texts]
        for r in results:
            save_to_db(r['text'], r['stress_level'], r['cause'])
        df_res = pd.DataFrame(results)
        st.dataframe(df_res[['text', 'stress_level', 'cause']])
        st.download_button("Download Results CSV", df_res.to_csv(index=False), "results.csv", "text/csv")
    except Exception as e:
        st.error(f"⚠️ Could not process the selected column. Error: {e}")



st.subheader("Try One Message/Search and Get Advice")
single = st.text_input("Type a message/query")
if st.button("Get Advice"):
    if single.strip():
        r = analyze_query(single)
        save_to_db(r['text'], r['stress_level'], r['cause'])
        st.write("Stress:", r['stress_level'], "| Cause:", r['cause'])
        st.info(ADVICE.get(r['cause'], "Take a short break!"))


st.subheader("🎤 Voice Input Stress Analyzer")

if st.button("Record Voice & Analyze"):
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Recording... Please speak now (10 sec max).")
            audio = r.listen(source, phrase_time_limit=10)
        text = r.recognize_google(audio)
        st.success(f"Detected text: {text}")
        
        
        result = analyze_query(text)
        save_to_db(result['text'], result['stress_level'], result['cause'])
        st.write("Stress Level:", result['stress_level'])
        st.write("Cause:", result['cause'])
        st.info(ADVICE.get(result['cause'], "Take a short break!"))

        
        engine = pyttsx3.init()
        engine.say(f"Stress detected: {result['stress_level']}. Advice: {ADVICE.get(result['cause'], 'Relax')}")
        engine.runAndWait()
    except Exception as e:
        st.error("Could not process audio. Try again.")


st.subheader("2️⃣ Stress Dashboard (All Records)")
df_all = load_all_records()
if not df_all.empty:
    
    st.write("Stress Level Distribution")
    st.bar_chart(df_all['stress_level'].value_counts())

    
    st.write("Cause Distribution")
    st.bar_chart(df_all['cause'].value_counts())

    
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    df_sorted = df_all.sort_values('timestamp')
    df_sorted['count'] = 1
    trend = df_sorted.groupby([pd.Grouper(key='timestamp', freq='D'), 'stress_level']).count()['count'].unstack(fill_value=0)
    st.write("Stress Trend Over Time")
    st.line_chart(trend)

    
    high_count = df_all['stress_level'].value_counts().get('High', 0)
    if high_count >= 3:
        st.error(f"Mentor Alert: {high_count} high-stress messages detected (anonymous)!")
    else:
        st.success("No mentor alerts currently.")

    
    high_texts = " ".join(df_all[df_all['stress_level']=='High']['text'].tolist())
    if high_texts.strip():
        st.write("Wordcloud of High-Stress Messages")
        wc = WordCloud(width=800, height=300, background_color="white").generate(high_texts)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)


st.subheader("3️⃣ Simulated Integration")
st.markdown("""
**Connect to Google Classroom / Moodle / Forum (Simulated)**  
This shows your project **can scale to real platforms**:
- Import posts/messages from Google Classroom CSV
- Analyze for stress levels and causes
- Provides live dashboard & alerts
""")

if st.button("Simulate Import from Google Classroom"):
    
    simulated_messages = [
        "I am stressed about the upcoming assignment deadline",
        "Feeling lonely in class, no friends to study with",
        "Exhausted after finishing the project",
        "Worried about the internship interview next week"
    ]
    
    st.info("Simulating import of messages from Google Classroom...")
    
    
    results = [analyze_query(msg) for msg in simulated_messages]
    for r in results:
        save_to_db(r['text'], r['stress_level'], r['cause'])
    
    df_res = pd.DataFrame(results)
    st.success("Simulated messages imported and analyzed successfully!")
    st.dataframe(df_res[['text', 'stress_level', 'cause']])
    st.download_button("Download Results CSV", df_res.to_csv(index=False), "google_classroom_results.csv", "text/csv")

