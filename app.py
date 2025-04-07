from flask import Flask, render_template, request
import os
import speech_recognition as sr
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load grammar model and vectorizer
model = joblib.load("models/grammar_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")  # Make sure this exists and is the one used during training

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', text=None, score=None)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return "No file uploaded", 400

    file = request.files['audio']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Convert audio to text using SpeechRecognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return render_template('index.html', text="Could not understand audio", score=None)
        except sr.RequestError:
            return render_template('index.html', text="Speech recognition service failed", score=None)

    # Preprocess the text using the same vectorizer used during training
    X_input = vectorizer.transform([text])  # Vectorize the transcribed text
    score = model.predict(X_input)[0]       # Predict grammar score

    return render_template('index.html', text=text, score=round(score, 2))

if __name__ == '__main__':
    app.run(debug=True)
