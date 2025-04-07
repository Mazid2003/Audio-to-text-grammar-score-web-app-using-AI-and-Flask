import os
import whisper
import pandas as pd
import numpy as np
import joblib
import language_tool_python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pydub import AudioSegment

# ✅ Set FFmpeg path (ensure it's valid)
AudioSegment.converter = r"C:\\Users\\USER\\Downloads\\ffmpeg-2025-03-31-git-35c091f4b7-full_build\\ffmpeg-2025-03-31-git-35c091f4b7-full_build\\bin\\ffmpeg.exe"

# ✅ Initialize Whisper model
whisper_model = whisper.load_model("base")

# ✅ Initialize grammar checker
tool = language_tool_python.LanguageTool('en-US')

# ✅ Path to training dataset (make sure this folder exists)
train_audio_folder = os.path.abspath("C:\\Users\\USER\\Desktop\\SHL assignment\\audios_train")

# ✅ Ensure 'models' folder exists
os.makedirs("models", exist_ok=True)

# ✅ Convert audio to WAV if needed
def convert_audio(input_path, output_format="wav"):
    try:
        output_path = os.path.splitext(input_path)[0] + "." + output_format
        if not os.path.exists(output_path):
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format=output_format)
        return output_path
    except Exception as e:
        print(f"❌ Error converting {input_path}: {e}")
        return None

# ✅ Transcribe using Whisper
def transcribe_audio(audio_path):
    try:
        converted_audio = convert_audio(audio_path)
        if not converted_audio or not os.path.exists(converted_audio):
            raise FileNotFoundError(f"Converted audio not found: {converted_audio}")
        result = whisper_model.transcribe(converted_audio)
        return result["text"]
    except Exception as e:
        print(f"❌ Error transcribing {audio_path}: {e}")
        return ""

# ✅ Count grammar issues
def check_grammar(text):
    matches = tool.check(text)
    return len(matches)

# ✅ Supported audio formats
supported_formats = [".wav", ".mp3", ".m4a", ".flac", ".aac", ".webm", ".ogg"]

# ✅ Collect audio files
audio_files = [
    os.path.join(train_audio_folder, f)
    for f in os.listdir(train_audio_folder)
    if os.path.splitext(f)[1].lower() in supported_formats
]

# ✅ Debug: print found audio files
print(f"🔍 Found {len(audio_files)} audio files in '{train_audio_folder}'")

# ✅ Process files
data = []
for audio_file in audio_files:
    print(f"🎧 Processing: {audio_file}")
    if not os.path.exists(audio_file):
        print(f"⚠️ File missing: {audio_file}")
        continue
    text = transcribe_audio(audio_file)
    if text:
        word_count = len(text.split())
        grammar_errors = check_grammar(text)
        grammar_score = max(100 - (grammar_errors / word_count * 100), 0) if word_count > 0 else 0
        data.append([text, word_count, grammar_errors, grammar_score])
    else:
        print(f"⚠️ Skipping due to empty transcription: {audio_file}")

# ✅ DataFrame from results
df = pd.DataFrame(data, columns=["Transcription", "Word Count", "Grammar Errors", "Grammar Score"])

# ✅ Proceed if we have valid data
if df.empty:
    print("❌ No valid transcriptions found. Please check your audio files.")
else:
    print(f"✅ Processed {len(df)} audio files successfully.")

    # ✅ Train model
    X = df[["Word Count", "Grammar Errors"]]
    y = df["Grammar Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Evaluate and save model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"📊 Model RMSE: {rmse:.2f}")

    model_path = "models/grammar_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
