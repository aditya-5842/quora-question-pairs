# best performing model is GBDT with TF-IDF vectors along with hand crafted features
# let's use these to build an api for finding the similarity between given two questions
print("importing libraries...\n")

import warnings
warnings.filterwarnings('ignore')
from flask import Flask, jsonify, request, redirect, url_for, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from featurizer import extract_features


print("loading models...\n")
d = "./models/"
with open(d+"std_tfidf.pkl", "rb") as f:
    std = pickle.load(f)
with open(d+"tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open(d+"tfidf_GBDT_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def inputs():
   return render_template('index.html')

@app.route('/output/', methods=["POST"])
def output():
    a="me"
    data = request.form.to_dict()
    q1 = data.get('q1')
    q2 = data.get('q2')
    prob = data.get('probabiliy')

    #convert it into dataframe
    new_df = pd.DataFrame(columns = ['question1','question2'])
    new_df = new_df.append({'question1': q1, 'question2':q2}, ignore_index = True)
    new_df = extract_features(new_df) #getting advance and basic features
    #get the tfidf vectorizer of text
    x_q1 = vectorizer.transform(new_df["question1"])
    x_q2 = vectorizer.transform(new_df["question2"])
    cols = [i for i in new_df.columns if i not in ['question1', 'question2']]
    new_df = new_df.loc[:,cols].values
    #get the hand crafted features
    X = hstack((x_q1, x_q2, new_df)).tocsr()
    X = std.transform(X)

    y_q = model.predict(X)
    y_q_proba = model.predict_proba(X)
    result = dict()
    result["Question-1"] = q1
    result["Question-2"] = q2

    if y_q == 1:
        result["Predicted Class"] = 'Similar'
    else:
        result["Predicted Class"] = "Not Similar"

    if prob=="yes":
        result["Probabiliy"] = round(max(y_q_proba[0]),4)

    return render_template('output.html', result = result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
