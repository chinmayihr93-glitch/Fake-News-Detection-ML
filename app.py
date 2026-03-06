from flask import Flask, render_template, request
import pickle
from preprocess import clean_text

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news_text']
    cleaned_text = clean_text(text)
    vector = tfidf.transform([cleaned_text])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])
    return render_template('index.html', 
                           prediction_text=f"News is {'Fake' if prediction==1 else 'Real'}",
                           confidence=f"Confidence: {confidence*100:.2f}%")

if __name__ == "__main__":
    app.run(debug=True)