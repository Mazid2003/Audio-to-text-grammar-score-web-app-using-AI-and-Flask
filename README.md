# 🗣️ Audio-to-text-grammar-score-web-app-using-AI-and-Flask

This web application allows users to upload or record audio, converts the speech to text using speech recognition, and provides a grammar score prediction using a trained machine learning model. The interface includes a toggle button for switching between light and dark mode for better accessibility and user experience.

---

## 🌟 Features

- 🎙️ **Speech-to-Text** conversion using audio files.
- ✅ **Grammar Score Prediction** using a custom ML model.
- 📁 Upload audio files directly.
- 🌗 **Dark Mode Toggle** for a visually pleasing experience.
- 💻 Responsive design compatible with all screen sizes.
- 📊 Displays both the predicted grammar score and the transcribed text.

---

## 📁 Project Structure

project/ 
├── app.py # Flask backend to handle audio and prediction 
├── models
    ├── grammar_model.pkl # Pre-trained grammar score prediction model
    ├── vectorizer.pkl
├── static/ 
    │ └── style.css # Custom CSS with dark mode support (if needed)
├── templates/ 
    │ └── index.html # HTML frontend with upload form & result display (you can directly include css as internal css)
├── audios_train/ # Training audio dataset (if applicable) 
├── audios_test/ # Test audio files for demo └── README.md # You're here!
├── uploads/ #audio files of the users(i.e, input files) will be stored in this folder 

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Flask
- scikit-learn
- SpeechRecognition
- pyaudio (for real-time recording if implemented)
- Any additional libraries used in your model pipeline

### Installation

1. **Clone the repository**

git clone https://github.com/Mazid2003/Audio-to-text-grammar-score-web-app-using-AI-and-Flask.git

cd directory

Install dependencies

pip install -r requirements.txt

Run the Flask App

python app.py

Open your browser and navigate to:

http://127.0.0.1:5000

**🧠 Model Details**

The model is trained on labeled audio data.

Audio is converted to text using SpeechRecognition.

Text is preprocessed and passed to a machine learning model (stored in model.pkl) that predicts a grammar score between 0 and 100.

**💡 Technologies Used**

Python 🐍

Flask 🔥

HTML5 & CSS3 🎨

JavaScript (for dark mode toggle)

scikit-learn 🤖

SpeechRecognition 🎙️

**📸 UI Preview**

Light Mode ☀️ vs Dark Mode 🌙
<img src="screenshots/light_mode.png" width="45%"/> <img src="screenshots/dark_mode.png" width="45%"/>

**🔐 License**

This project is licensed under the MIT License.

**🙋‍♂️ Author**

**Mohammad Mazid**

📧 mazidmd750@gmail.com


**🤝 Contributing**

Contributions, suggestions, and improvements are welcome!
Feel free to fork this project and submit a PR 🚀

