import os
import joblib
from flask import Flask, render_template, request, send_from_directory, render_template_string, jsonify
from werkzeug.utils import secure_filename
from nlp_model import NLPModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'filename': filename})
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/delete_file', methods=['POST'])
def delete_file():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    This is the API endpoint for the NLP prediction.
    It receives text from the user, processes it, and returns a response.
    """
    try:
        user_message = request.form.get('message', '').lower().strip()
        mode = request.form.get('mode', 'sentiment').lower().strip()
        filename = request.form.get('filename')

        if not user_message:
            return get_error_template('No message or file provided.')

        nlp_model = NLPModel()
        nlp_result = nlp_model.predict_answer(user_message, mode, filename)

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
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
