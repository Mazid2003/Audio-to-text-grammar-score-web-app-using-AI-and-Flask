from flask import Flask, render_template, request
import os
import speech_recognition as sr
import joblib
from werkzeug.utils import secure_filename
from pydub import AudioSegment

app = Flask(__name__)

# Load grammar model and vectorizer
model = joblib.load("models/grammar_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', text=None, score=None)

# Route for uploaded audio files
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

    return handle_audio_file(file_path)

# Route for recorded audio (assumed webm)
@app.route('/recorded-audio', methods=['POST'])
def recorded_audio():
    if 'audio' not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files['audio']
    if file.filename == '':
        return {"error": "No selected file"}, 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Convert from webm to wav using pydub
    try:
        audio_webm = AudioSegment.from_file(file_path, format="webm")
        wav_path = file_path.replace(".webm", ".wav")
        audio_webm.export(wav_path, format="wav")
    except Exception as e:
        return {"error": f"Audio conversion failed: {str(e)}"}, 500

    return handle_audio_file(wav_path, json_output=True)

# Common handler for audio processing
def handle_audio_file(path, json_output=False):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = "Could not understand audio"
        score = None
    except sr.RequestError:
        text = "Speech recognition failed"
        score = None
    else:
        X_input = vectorizer.transform([text])
        score = model.predict(X_input)[0]
        score = round(score, 2)

    if json_output:
        return {"transcript": text, "grammar_score": score}
    return render_template('index.html', text=text, score=score)

if __name__ == '__main__':
    app.run(debug=True)
