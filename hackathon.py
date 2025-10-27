import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import speech_recognition as sr
import pyttsx3

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

# ------------------ Streamlit Layout ------------------
st.set_page_config(page_title="MindTrack Chatbot", layout="wide")
st.title("MindTrack: AI-Powered Academic Stress Chatbot")
st.markdown("**Choose an option from the sidebar to interact. Analysis results and visualizations appear below.**")

# ------------------ Sidebar Feature Selection ------------------
feature = st.sidebar.radio("Choose Feature", ["💬 Text Input", "🎤 Voice Input", "📁 CSV Upload", "📥 Simulated Google Classroom"])

# ------------------ Text Input ------------------
if feature == "💬 Text Input":
    st.subheader("💬 Chat with MindTrack")
    user_msg = st.text_input("Type your message")
    if st.button("Analyze Message", key="text_input"):
        if user_msg.strip():
            result = analyze_query(user_msg)
            save_to_db(result['text'], result['stress_level'], result['cause'])
            # Display like a chat
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"**MindTrack:** Stress: {result['stress_level']} | Cause: {result['cause']}")
            st.info(ADVICE.get(result['cause']))

# ------------------ Voice Input ------------------
elif feature == "🎤 Voice Input":
    st.subheader("🎤 Voice Input")
    if st.button("Click to Record & Analyze", key="voice_input"):
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                st.info("Recording (10 sec max)... Speak now!")
                audio = r.listen(source, phrase_time_limit=10)
            text = r.recognize_google(audio)
            result = analyze_query(text)
            save_to_db(result['text'], result['stress_level'], result['cause'])
            st.markdown(f"**You (voice):** {text}")
            st.markdown(f"**MindTrack:** Stress: {result['stress_level']} | Cause: {result['cause']}")
            st.info(ADVICE.get(result['cause']))
            # Voice feedback
            engine = pyttsx3.init()
            engine.say(f"Stress detected: {result['stress_level']}. Advice: {ADVICE.get(result['cause'])}")
            engine.runAndWait()
        except:
            st.error("Could not process audio. Try again.")

# ------------------ CSV Upload ------------------
elif feature == "📁 CSV Upload":
    st.subheader("📁 Upload CSV for Batch Analysis")
    uploaded = st.file_uploader("Upload CSV with messages", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        text_col = None
        for col in df_in.columns:
            if any(x in col.lower() for x in ["query", "message", "text", "response"]):
                text_col = col
                break
        if not text_col:
            text_col = st.selectbox("Select the text column:", df_in.columns)
        try:
            texts = df_in[text_col].astype(str).tolist()
            results = [analyze_query(t) for t in texts]
            for r in results:
                save_to_db(r['text'], r['stress_level'], r['cause'])
            df_res = pd.DataFrame(results)
            st.success("CSV analyzed successfully!")
            st.dataframe(df_res[['text','stress_level','cause']])
            st.download_button("Download Results CSV", df_res.to_csv(index=False), "batch_results.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Could not process the CSV. Error: {e}")

# ------------------ Simulated Google Classroom ------------------
elif feature == "📥 Simulated Google Classroom":
    st.subheader("📥 Simulated Google Classroom Import")
    if st.button("Simulate Import", key="google_import"):
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
        st.download_button("Download Results CSV", df_res.to_csv(index=False),
                           "google_classroom_results.csv", "text/csv")

# ------------------ Dashboard Visualizations (Always Visible) ------------------
st.subheader("📊 Stress Dashboard (Live)")
df_all = load_all_records()
if not df_all.empty:
    st.write("**Stress Level Distribution**")
    st.bar_chart(df_all['stress_level'].value_counts())

    st.write("**Cause Distribution**")
    st.bar_chart(df_all['cause'].value_counts())

    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    df_sorted = df_all.sort_values('timestamp')
    df_sorted['count'] = 1
    trend = df_sorted.groupby([pd.Grouper(key='timestamp', freq='D'), 'stress_level']).count()['count'].unstack(fill_value=0)
    st.write("**Stress Trend Over Time**")
    st.line_chart(trend)

    # Mentor Alerts
    high_count = df_all['stress_level'].value_counts().get('High', 0)
    if high_count >= 3:
        st.error(f"Mentor Alert: {high_count} high-stress messages detected!")
    else:
        st.success("No mentor alerts currently.")

    # Wordcloud
    high_texts = " ".join(df_all[df_all['stress_level']=='High']['text'].tolist())
    if high_texts.strip():
        st.write("**Wordcloud of High-Stress Messages**")
        wc = WordCloud(width=800, height=300, background_color="white").generate(high_texts)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
