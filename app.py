from flask import Flask, request, render_template
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

STOPWORDS = set(stopwords.words("english"))

# Load the models 
clf = pickle.load(open('model/model_svm.pkl', 'rb'))
cv = pickle.load(open('model/countVectorizer.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        # Preprocess the message
        stemmer = PorterStemmer()
        review = re.sub("[^a-zA-Z]", " ", message)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        
        # Transform the input using the CountVectorizer
        vect = cv.transform([review]).toarray()

        # Scale the transformed input
        vect_scaled = scaler.transform(vect)

        # Make the prediction
        my_prediction = clf.predict(vect_scaled)

        # Convert the prediction to a readable format
        prediction_label = "Positive" if my_prediction[0] == 1 else "Negative"

        return render_template('result.html', prediction=prediction_label)

if __name__ == "__main__":
    app.run(host="localhost", port=5900)
