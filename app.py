# app.py
import streamlit as st
import pandas as pd
from sentiment_model import predict_sentiment

# Store past predictions
if 'history' not in st.session_state:
    st.session_state.history = []

# UI Layout
st.title("✈️ Airline Tweet Sentiment Analyzer")

# Dropdown of airlines
airlines = ['United', 'US Airways', 'American', 'Southwest', 'Delta', 'Virgin America']
airline = st.selectbox("Select Airline", airlines)

# Input tweet
tweet = st.text_area("Enter your tweet", placeholder="Type your tweet here...")

# Predict button
if st.button("Analyze Sentiment"):
    sentiment = predict_sentiment(tweet)
    st.success(f"Predicted Sentiment for {airline}: **{sentiment}**")

    # Save in session state
    st.session_state.history.append({
        "Airline": airline,
        "Tweet": tweet,
        "Sentiment": sentiment
    })

# Show past predictions
if st.session_state.history:
    st.write("### Previous Predictions")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
