import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained model from the file
with open('logistic_regression_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the same vectorizer used during training
with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title("Sentiment Analysis App")

# Text input for user to enter text
text = st.text_input("Enter a text for sentiment analysis:", "This product is best")

# Make predictions when a user clicks a button
if st.button("Predict Sentiment"):
    # Preprocess the user's input using the same vectorizer
    text_features = vectorizer.transform([text])

    # Make a prediction using the loaded model
    predicted_sentiment = classifier.predict(text_features)

    # Assuming that your model is binary (positive/negative sentiment)
    if predicted_sentiment[0] == 1:
        sentiment = "positive"
    else:
        sentiment = "negative"

    st.write(f"The sentiment of the text is {sentiment}")
