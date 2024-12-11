from flask import Flask, request, jsonify
import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import json
import pickle  # Import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = load_model('model.h5')

# Load the intents JSON and word/label pickle files
intents = json.loads(open('D:/PROJECT/intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Predict intent function
def predict_intent(text):
    words_in_text = nltk.word_tokenize(text.lower())
    words_in_text = [lemmatizer.lemmatize(w) for w in words_in_text]
    
    # Create the bag of words
    bag = [1 if w in words_in_text else 0 for w in words]
    
    # Reshape to match the model's input shape
    bag = np.array(bag).reshape(1, -1)
    
    # Predict the intent
    prediction = model.predict(bag)[0]
    intent_index = np.argmax(prediction)
    
    # If prediction is above 0.5, return the class
    return classes[intent_index] if prediction[intent_index] > 0.5 else None

# Get response for the predicted intent
def get_response(intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I'm not sure how to respond to that."

# Create the API route to handle the chatbot prediction
@app.route('/chat', methods=['POST'])
def chat():
    # Get user input from the request
    user_input = request.json.get('message')
    
    # Get the predicted intent and response
    intent = predict_intent(user_input)
    if intent:
        response = get_response(intent)
    else:
        response = "I don't understand."
    
    # Return the response in JSON format
    return jsonify({"response": response})

# Run the Flask app
# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

