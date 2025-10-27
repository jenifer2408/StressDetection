import random
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import speech_recognition as sr
import pyttsx3
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os


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
    wc = WordCloud(width=600, height=300, background_color="white", colormap='plasma').generate(" ".join(df_session['text']))
    fig, ax = plt.subplots(figsize=(8,3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ------------------ Streamlit Layout ------------------
st.set_page_config(page_title="MindTrack Chatbot", layout="wide")
st.markdown("<h1 style='color:#4B0082;'>🧠 MindTrack: AI-Powered Stress Chatbot</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='color:#FF4500; font-size:16px'>
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
    st.markdown("<h2 style='color:#1E90FF;'>💬 Chat with MindTrack</h2>", unsafe_allow_html=True)
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
    st.markdown("<h2 style='color:#1E90FF;'>🎤 Voice Input</h2>", unsafe_allow_html=True)
    if st.button("Click to Record & Analyze", key="voice_input"):
        r = sr.Recognizer()
        try:
            st.info("Recording (10 sec max)... Speak now!")
            fs = 44100  # Sample rate
            duration = 10  # seconds
            recording = sd.rec( int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()

# Save temp file and process it
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            write(tmp_wav.name, fs, recording)

            with sr.AudioFile(tmp_wav.name) as source:
                audio = r.record(source)

            text = r.recognize_google(audio)
            os.remove(tmp_wav.name)

            result = analyze_query(text)
            save_to_db(result['text'], result['stress_level'], result['cause'])
            st.markdown(f"**You (voice):** {text}")
            st.markdown(f"**MindTrack:** Stress: {result['stress_level']} | Cause: {result['cause']}")
            st.info(ADVICE.get(result['cause']))

            engine = pyttsx3.init()
            engine.say(f"Stress detected: {result['stress_level']}. Advice: {ADVICE.get(result['cause'])}")
            engine.runAndWait()

            session_df = pd.DataFrame([result])
            display_session_graphs(session_df)
        except:
            st.error("Could not process audio. Try again.")

# ------------------ CSV Upload ------------------
elif feature == "📁 CSV Upload":
    st.markdown("<h2 style='color:#32CD32;'>📁 Upload CSV for Batch Analysis</h2>", unsafe_allow_html=True)
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

            display_session_graphs(df_res)
        except Exception as e:
            st.error(f"⚠️ Could not process the CSV. Error: {e}")

# ------------------ Simulated Google Classroom ------------------
elif feature == "📥 Simulated Google Classroom":
    st.markdown("<h2 style='color:#FF8C00;'>📥 Simulated Google Classroom Import</h2>", unsafe_allow_html=True)
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

        display_session_graphs(df_res)

# ------------------ MindCheck: Stress Detective ------------------
elif feature == "🕵️ MindCheck: Stress Detective":
    st.markdown("<h2 style='color:#8A2BE2;'>🕵️ MindCheck: Stress Detective</h2>", unsafe_allow_html=True)
    st.markdown("Answer these situational questions and MindTrack will detect your stress level and cause!")

    questions = [
        {"q": "How often do you feel overwhelmed by assignments?", "options": ["Never", "Sometimes", "Often", "Always"], "weights": [0,1,2,3]},
        {"q": "Do you feel lonely or disconnected in class?", "options": ["Never", "Sometimes", "Often", "Always"], "weights": [0,1,2,3]},
        {"q": "Are you feeling exhausted or burned out lately?", "options": ["Never", "Sometimes", "Often", "Always"], "weights": [0,1,2,3]},
        {"q": "Are you worried about career or placement?", "options": ["Never", "Sometimes", "Often", "Always"], "weights": [0,1,2,3]}
    ]

    answers = []
    total_score = 0
    for i, item in enumerate(questions):
        ans = st.radio(item["q"], item["options"], key=f"mcq{i}")
        score = item["weights"][item["options"].index(ans)]
        answers.append({"question": item["q"], "answer": ans, "score": score})
        total_score += score

    if st.button("Get MindCheck Result", key="mindcheck"):
        if total_score <= 3:
            level = "Low"
        elif total_score <= 6:
            level = "Medium"
        else:
            level = "High"

        cause = max(["Academic","Social","Burnout","Career"], key=lambda x: random.randint(0,3)) # Randomly assign for demo
        st.markdown(f"**Stress Level:** {level}")
        st.markdown(f"**Likely Cause:** {cause}")
        st.info(ADVICE.get(cause))
