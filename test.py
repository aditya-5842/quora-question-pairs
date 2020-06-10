print("importing libraries...\n")
import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from featurizer import extract_features
import warnings
warnings.filterwarnings('ignore')

#let's load models which are required for prediction
# GBDT with tfidf vectorizer gave the best performance,
#so laoding saved things are which are required for this
print("loading models...\n")
d = "./Models/"
with open(d+"std_tfidf.pkl", "rb") as f:
    std = pickle.load(f)
with open(d+"tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open(d+"tfidf_GBDT_model.pkl", "rb") as f:
    model = pickle.load(f)

#let's take the input from users for prediction
more_input = True
while more_input:
    new_df = pd.DataFrame(columns = ['question1','question2'])
    print('Write first question:', end = " ")
    q1 = input()
    print('Write seconde question:', end = " ")
    q2 = input()

    print("\nvectorizing data...")
    #convert it into dataframe
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
    print('\nPredicted class is: {} i.e. {} and'.format(y_q, "Similar" if y_q == 1 else "Not Similar"))
    print("Probability of predicted class is {:.4f}".format(max(y_q_proba[0])))

    print("\nDo you want to check more: Pess 1 if yes, Ohterwise it'll terminate the session.")
    try:
        q3 = int(input())
        if q3 == 1:
            more_input = True
            print()
        else:
            more_input = False
    except:
        more_input = False
