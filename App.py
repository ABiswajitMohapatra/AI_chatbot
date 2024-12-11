import streamlit as st
import nltk
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

# Load the model and data
@st.cache_resource
def load_model_and_data():
    model = load_model('flask/model.h5')
    intents = json.loads(open('flask/intents.json').read())
    words = pickle.load(open('flask/texts.pkl', 'rb'))
    classes = pickle.load(open('flask/labels.pkl', 'rb'))
    return model, intents, words, classes

model, intents, words, classes = load_model_and_data()

# Prediction function
def predict_intent(text):
    words_in_text = nltk.word_tokenize(text.lower())
    words_in_text = [lemmatizer.lemmatize(w) for w in words_in_text]
    bag = [1 if w in words_in_text else 0 for w in words]
    bag = np.array(bag).reshape(1, -1)
    prediction = model.predict(bag)[0]
    intent_index = np.argmax(prediction)
    return classes[intent_index] if prediction[intent_index] > 0.5 else None

# Get response function
def get_response(intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I'm not sure how to respond to that."

# Streamlit UI
st.title("AI Chatbot 🤖")
st.write("Welcome! I'm here to help you. Type a message below to start chatting.")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Check if the user wants to quit
    if prompt.lower() == "quit":
        st.session_state.messages.append({"role": "assistant", "content": "Goodbye! Have a great day!"})
        with st.chat_message("assistant"):
            st.write("Goodbye! Have a great day!")
        st.stop()  # Stop further execution to exit the conversation

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get chatbot response
    intent = predict_intent(prompt)
    if intent:
        response = get_response(intent)
    else:
        response = "I don't understand. Could you please rephrase that?"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Add some styling
st.markdown("""
    <style>
    .stChat {
        padding: 20px;
        border-radius: 10px;
    }
    .stChatMessage {
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

