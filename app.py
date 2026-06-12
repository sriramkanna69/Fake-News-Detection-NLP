import streamlit as st
import pickle

# Load Model
model = pickle.load(open("fake_news_model.pkl", "rb"))

# Load Vectorizer
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

st.title("📰 Fake News Detection")

news = st.text_area("Enter News Article")

if st.button("Check News"):

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        st.success("REAL NEWS")
    else:
        st.error("FAKE NEWS")