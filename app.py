import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import joblib

df = pd.read_csv(r"Sentiment Analysis Dataset.csv", encoding='ISO-8859-1')

df.dropna(subset=['SentimentText'], inplace=True)

print(df['Sentiment'].value_counts())

df['Sentiment'] = df['Sentiment'].replace({0: 'negative', 1: 'positive'})
df = df.sample(20000, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(df['SentimentText'], df['Sentiment'], test_size=0.2, random_state=42)

# Create a pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', MultinomialNB())
])

model.fit(X_train, y_train)

joblib.dump(model, 'sentiment_chatbot_model.pkl')

print("Model trained and saved!")

import streamlit as st

model = joblib.load('sentiment_chatbot_model.pkl')

st.title("🧠 ML-Powered Mental Health Chatbot")

# Add examples
examples = [
    "I am feeling so great today!",
    "I feel really sad, everything seems wrong.",
    "I'm just trying to get through the day.",
    "I don’t know how I feel right now.",
    "I’m so excited about everything today!"
]

example = st.selectbox("Or try one of these examples:", examples)

user_input = st.text_input("You:")

if user_input:
    prediction = model.predict([user_input])[0]
    st.markdown(f"**Predicted Sentiment:** `{prediction}`")

    if prediction == "negative":
        response = "I'm really sorry you're feeling that way. You're not alone. It's okay to feel this way. 💙"
    elif prediction == "positive":
        response = "That's wonderful to hear! Keep holding onto that positivity. 🌟"
    else:
        response = "Thank you for sharing. I'm here to chat anytime."

    st.text_area("Bot:", value=response, height=150)
