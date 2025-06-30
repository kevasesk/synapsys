import os
import joblib
from flask import Flask, render_template, request, jsonify
from nlp_model import NLPModel

app = Flask(__name__)

def nlp_prediction(user_message):
    user_message = user_message.lower().strip()
    nlp_model = NLPModel()
    
    return nlp_model.predict_message(user_message)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    This is the API endpoint for the NLP prediction.
    It receives text from the user, processes it, and returns a response.
    """
    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400


        probability, bot_response = nlp_prediction(user_message)

        return jsonify({'answer': bot_response, 'probability': probability})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An internal server error occurred'}), 500


if __name__ == '__main__':
    app.run(debug=True)
