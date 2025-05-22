import streamlit as st
from analysis import parse_transcript, compute_sentiment, compute_filler_ratio, compute_word_count
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Dialogue Analysis Dashboard", layout="wide")

st.title("Dialogue Analysis Dashboard")

# Load transcript
turns = parse_transcript("transcript.txt")

# Compute metrics for each turn
sentiments = []
filler_ratios = []
word_counts = []
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
for turn in turns:
    text = turn["text"]
    sentiments.append(compute_sentiment(text))
    filler_ratios.append(compute_filler_ratio(text))
    word_counts.append(compute_word_count(text))

# Prepare DataFrame for display
df = pd.DataFrame({
    "Speaker": [t["speaker"] for t in turns],
    "Text": [t["text"] for t in turns],
    "Sentiment": sentiments,
    "Filler Ratio": filler_ratios,
    "Word Count": word_counts
})

# Display transcript and metrics
st.dataframe(df, use_container_width=True)

# Show overall averages
st.subheader("Overall Averages")
col1, col2 = st.columns(2)
col1.metric("Average Filler Ratio", f"{df['Filler Ratio'].mean():.2%}")
col2.metric("Average Word Count", f"{df['Word Count'].mean():.1f}")

# Creative extension: Word count bar chart
st.subheader("Word Count per Turn")
st.bar_chart(df["Word Count"])

# Optionally, show sentiment distribution
st.subheader("Sentiment Distribution")
st.bar_chart(df["Sentiment"].value_counts()) 