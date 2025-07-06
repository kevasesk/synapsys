import os
import joblib
from flask import Flask, render_template, request, send_from_directory, render_template_string
from nlp_model import NLPModel

app = Flask(__name__)

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
        user_message = request.form.get('message', '').lower().strip()
        mode = request.form.get('mode', 'sentiment').lower().strip()

        if not user_message:
            return get_error_template('No message provided. Please enter a message to analyze.')

        nlp_model = NLPModel()
        nlp_result = nlp_model.predict_answer(user_message, mode)

        if not nlp_result:
            return get_error_template('No such mode for')

        return render_template('bot_response.html', mode=mode.capitalize(), nlp_result=nlp_result)

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


@app.route('/films/<path:filename>')
def serve_film(filename):
    return send_from_directory('films', filename)


if __name__ == '__main__':
    app.run(debug=True)
