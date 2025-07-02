import os
import joblib
from flask import Flask, render_template, request, send_from_directory, render_template_string
from nlp_model import NLPModel

app = Flask(__name__)

@app.route('/films/<path:filename>')
def serve_film(filename):
    return send_from_directory('films', filename)

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
        user_message = request.form.get('message')

        if not user_message:
            return get_error_template('No message provided. Please enter a message to analyze.')


        sentiment_data, spam_data = nlp_prediction(user_message)

        return render_template('bot_response.html', sentiment_data=sentiment_data, spam_data=spam_data)

    except Exception as e:
          return get_error_template(str(e))


def get_error_template(message):
    error_template = """
        <div class="flex justify-start">
            <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-lg max-w-xs">
                <p class="font-bold">Error</p>
                 <p>An error occurred: {{ message }}</p>
            </div>
        </div>
        """
    return render_template_string(error_template), 400

if __name__ == '__main__':
    app.run(debug=True)
