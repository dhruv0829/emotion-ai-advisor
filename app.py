from flask import Flask, render_template, request
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import csv
from datetime import datetime


# Download only first time
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Load saved model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
emotion_data = {
    "joy": {
        "emoji": "😊",
        "color": "#FFD93D",
        "advice": [
            "Keep spreading positivity around you.",
            "Celebrate this joyful moment fully.",
            "Write down what made you happy today.",
            "Share your happiness with someone."
        ]
    },
    "sadness": {
        "emoji": "😢",
        "color": "#5DADE2",
        "advice": [
            "It’s okay to feel sad. Allow yourself to process emotions.",
            "Talk to someone you trust.",
            "Take a short walk in nature.",
            "Write your thoughts in a journal."
        ]
    },
    "anger": {
        "emoji": "😡",
        "color": "#E74C3C",
        "advice": [
            "Pause and take slow deep breaths.",
            "Step away before reacting.",
            "Try exercising to release tension.",
            "Listen to calming music."
        ]
    },
    "fear": {
        "emoji": "😨",
        "color": "#AF7AC5",
        "advice": [
            "Focus only on what you can control.",
            "Practice grounding techniques.",
            "Break problems into smaller steps.",
            "Talk about your worries openly."
        ]
    },
    "love": {
        "emoji": "❤️",
        "color": "#FF6B81",
        "advice": [
            "Express your appreciation openly.",
            "Spend meaningful time with loved ones.",
            "Write a gratitude message.",
            "Nurture strong relationships."
        ]
    }
}




def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    cleaned = []
    for word in tokens:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            cleaned.append(lemma)

    return " ".join(cleaned)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    advice = None
    emoji = None
    color = "#333"
    history = []

    if request.method == "POST":
        user_input = request.form["text"]
        processed = preprocess(user_input)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0].lower()

        if prediction in emotion_data:
            emoji = emotion_data[prediction]["emoji"]
            color = emotion_data[prediction]["color"]
            advice = random.choice(emotion_data[prediction]["advice"])
        else:
            emoji = "🙂"
            advice = "Take care of yourself."
            color = "#333"

        # Save history
        with open("history.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), user_input, prediction])

    # Load history
    try:
        with open("history.csv", "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            history = reader[-5:]
    except:
        history = []

    return render_template("index.html",
                           prediction=prediction,
                           advice=advice,
                           emoji=emoji,
                           color=color,
                           history=history)


if __name__ == "__main__":
    app.run()
import os 
if __name__=="__main__":
    port =int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)