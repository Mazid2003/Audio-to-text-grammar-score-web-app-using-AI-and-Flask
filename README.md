# 🗣️ Audio-to-text-grammar-score-web-app-using-AI- Python-and-Flask

This web application allows users to upload or record audio, converts the speech to text using speech recognition, and provides a grammar score prediction using a trained machine learning model. The interface includes a toggle button for switching between light and dark mode for better accessibility and user experience.

---
**you can also visit the website**

**https://aigrammaranalyzer.pythonanywhere.com**

## 🌟 Features

- 🎙️ **Speech-to-Text** conversion using audio files.
- ✅ **Grammar Score Prediction** using a custom ML model.
- 📁 Upload audio files directly.
- 🌗 **Dark Mode Toggle** for a visually pleasing experience.
- 💻 Responsive design compatible with all screen sizes.
- 📊 Displays both the predicted grammar score and the transcribed text.

---

## 📁 Project Structure
```
project/ 
├── app.py # Flask backend to handle audio and prediction 
├── models
    ├── grammar_model.pkl # trained grammar score prediction model
    ├── vectorizer.pkl
├── static/ 
    │ └── style.css # Custom CSS with dark mode support (if needed)
├── templates/ 
    │ └── index.html # HTML frontend with upload form & result display (you can directly include css as internal css)
├── audios_train/ # Training audio dataset  
├── audios_test/ # Test audio files for demo 
├── uploads/ #audio files of the users(i.e, input files) will be stored in this folder
├── train_model.py/ # for training the grammar_model.pkl file 
├── train_model.ipynb/ # for training the vectorizer.pkl file
```
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

Light Mode ☀️  

![out1](https://github.com/user-attachments/assets/7892778c-1437-43fa-bed4-dbb868746d9c)

![out5](https://github.com/user-attachments/assets/bfc5f53e-9244-4a9b-a5dd-a346d8b68c7e)

Dark Mode 🌙

![out2](https://github.com/user-attachments/assets/02cd2240-a914-4cdb-b7ec-00f1f67b88ed)

![out4](https://github.com/user-attachments/assets/cc84ae1d-e27c-41f9-888b-23dd57793d61)

**🔐 License**

This project is licensed under the MIT License.

**🙋‍♂️ Author**

**Mohammad Mazid**

📧 mazidmd750@gmail.com


**🤝 Contributing**

Contributions, suggestions, and improvements are welcome!
Feel free to fork this project and submit a PR 🚀

