# sentiment_analyzer.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🧠")

st.title("🧠 AI Sentiment Analyzer")
st.write("Analyze the sentiment of text using a simple AI model (Positive, Negative, or Neutral).")

# --- Step 1: Prepare dataset ---
data = {
    'text': [
        'I love this movie!', 'This was a terrible experience.',
        'Absolutely fantastic!', 'Worst service ever.',
        'It was okay, not great.', 'I am so happy today!',
        'I hate waiting in line.', 'The product quality is excellent!',
        'This is disappointing.', 'I feel amazing!',
        'The weather is fine.', 'I am not sure about this.',
        'What a waste of time.', 'Such a wonderful surprise!',
        'Mediocre performance overall.'
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative',
        'neutral', 'positive', 'negative', 'positive',
        'negative', 'positive', 'neutral', 'neutral',
        'negative', 'positive', 'neutral'
    ]
}

df = pd.DataFrame(data)

# --- Step 2: Train model (lightweight) ---
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))

st.sidebar.header("📊 Model Info")
st.sidebar.write(f"**Model:** Logistic Regression")
st.sidebar.write(f"**Accuracy:** {accuracy*100:.2f}%")

# --- Step 3: User input ---
user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        if prediction == "positive":
            st.success("✅ Sentiment: Positive 😊")
        elif prediction == "negative":
            st.error("❌ Sentiment: Negative 😠")
        else:
            st.info("😐 Sentiment: Neutral")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn.")
