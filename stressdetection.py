import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Download VADER only once
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')

# Initialize analyzer
sia = SentimentIntensityAnalyzer()

# ---------- DATABASE SETUP ----------
DB_PATH = "stress_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stress_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT,
        sentiment TEXT,
        score REAL,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

def save_to_db(message, sentiment, score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO stress_entries (message, sentiment, score, timestamp) VALUES (?, ?, ?, ?)",
              (message, sentiment, score, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_all_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM stress_entries", conn)
    conn.close()
    return df

# Initialize DB
init_db()

# ---------- STREAMLIT UI ----------
st.title("🧠 MindTrack: Student Stress Detection Dashboard")

st.sidebar.header("Choose Input Mode")
mode = st.sidebar.radio("Input Type:", ["Manual Entry", "Upload CSV"])

# ---------- MODE 1: MANUAL ENTRY ----------
if mode == "Manual Entry":
    user_message = st.text_area("Enter your message about how you feel:", "")
    if st.button("Analyze"):
        if user_message.strip():
            sentiment_score = sia.polarity_scores(user_message)
            compound = sentiment_score['compound']

            if compound >= 0.05:
                sentiment = "Positive"
            elif compound <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            save_to_db(user_message, sentiment, compound)
            st.success(f"✅ Sentiment Detected: {sentiment} ({compound})")

        else:
            st.warning("Please enter a message before analyzing.")

# ---------- MODE 2: UPLOAD CSV ----------
elif mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV (e.g., Google Form responses)", type=['csv'])
    if uploaded_file:
        df_in = pd.read_csv(uploaded_file)

        # Auto-detect relevant text column
        possible_cols = [c for c in df_in.columns if 'describe' in c.lower() or 'moment' in c.lower() or 'stress' in c.lower()]
        if possible_cols:
            text_col = possible_cols[0]
        else:
            text_col = df_in.columns[-1]

        st.info(f"Detected text column: **{text_col}**")

        texts = df_in[text_col].astype(str).fillna("").tolist()

        results = []
        for text in texts:
            score = sia.polarity_scores(text)['compound']
            sentiment = "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
            save_to_db(text, sentiment, score)
            results.append((text, sentiment, score))

        st.success("All responses analyzed and saved successfully!")

# ---------- SHOW DATABASE ----------
st.subheader("📊 Stored Stress Entries (From All Sessions)")
df_all = fetch_all_data()

if not df_all.empty:
    st.dataframe(df_all)

    # ---------- WORDCLOUD ----------
    st.subheader("☁️ WordCloud of Recent Responses")
    text_data = " ".join(df_all['message'].astype(str))
    if text_data.strip():
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
else:
    st.info("No data yet! Start by adding a message or uploading a CSV.")
