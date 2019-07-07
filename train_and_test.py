import numpy as np
import pandas as pd
import seaborn as sns
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from subprocess import check_output
import plotly.graph_objs as go
import plotly.tools as tls
import os, gc, re, distance
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.classification import accuracy_score, log_loss
from tqdm import tqdm
import spacy
from scipy.sparse import hstack
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

from feature_extraction_functions import extract_features, vectorize_question, get_TFIDF_W2V

# Read the data
df = pd.read_csv('train.csv')
print('Shape of data is: ', df.shape)
# to check for NaN value
print('There are {} rows which contain NaN values'.format(df.isnull().values.sum()))
print('Dropping the rows which have NaN...')
df.dropna(axis=0, how='any', inplace = True)
print('Now shape of data is: ', df.shape)

if os.path.isfile('extracted_features.csv'):
    df = pd.read_csv('extracted_features.csv', encoding = 'latin-1')
    print("Shape of Extracted Features is: ", df.shape)
else:
    df = extract_features(df)
    df.to_csv('extracted_features.csv', index = False)
    print('After Extraction shape of data is: ', df.shape)

# split the data
cols = [i for i in df.columns if i not in ['qid1', 'qid2', 'is_duplicate']]
X = df.loc[:,cols]
Y = df.loc[:, 'is_duplicate']
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify = Y, random_state=42)
print('Datapoints in data is: {} and after splitting datapoints in train is: {} and in test is: {}'\
.format(df.shape[0], x_train.shape[0], x_test.shape[0]))

# for TF-IDF with max_features = 5000
#I'll split the x_train into tfidf_train, tfidf_CV
tfidf_y_test = y_test
tfidf_x_test = x_test
tfidf_x_train, tfidf_cv, tfidf_y_train, tfidf_y_cv = train_test_split(x_train, y_train, test_size = 0.2,
                                                                      stratify = y_train, random_state = 42)
# I'll fit TF-IDF vectorizer with combined question1 and question2 of tfidf_x_train
# then cv and test data will be transformed using above dictionary
questions = list(tfidf_x_train['question1']) + list(tfidf_x_train['question2'])
tfidf1 = TfidfVectorizer(max_features = 5000, lowercase=True)
tfidf1.fit(questions)
#for train data
tfidf_train_q1, tfidf_train_q2 = tfidf1.transform(tfidf_x_train['question1']), tfidf1.transform(tfidf_x_train['question2'])
tfidf_cv_q1, tfidf_cv_q2 = tfidf1.transform(tfidf_cv['question1']), tfidf1.transform(tfidf_cv['question2'])
tfidf_test_q1, tfidf_test_q2 = tfidf1.transform(tfidf_x_test['question1']), tfidf1.transform(tfidf_x_test['question2'])
# for TFIDF merging
tfidf_x_train = hstack((tfidf_train_q1, tfidf_train_q1, tfidf_x_train.iloc[:,3:])) #tfidf_y_train
tfidf_x_cv = hstack((tfidf_cv_q1, tfidf_cv_q2, tfidf_cv.iloc[:,3:]))              #tfidf_y_cv
tfidf_x_test = hstack((tfidf_test_q1, tfidf_test_q2, tfidf_x_test.iloc[:,3:]))    #tfidf_y_test/y_test


# Getting the IDF score of each word
questions = list(x_train['question1']) + list(x_train['question2'])
tfidf2 = TfidfVectorizer(lowercase=True)
tfidf2.fit(questions)
# dict key:word and value:tf-idf score
word_to_idf_score = dict(zip(tfidf2.get_feature_names(), tfidf2.idf_))
print('Lenght of features of tf-idf vector: ', len(tfidf2.get_feature_names()))
#getting TF-IDF W2V vector
tfidf_w2v_q1_train, tfidf_w2v_q2_train, tfidf_w2v_q1_test, tfidf_w2v_q2_test = get_TFIDF_W2V(x_train, x_test, word_to_idf_score)
# #Now let's merge or stack the data using np.hstack(): https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html
cols = [i for i in x_train.columns if i not in ['id', 'qid1', 'qid2','question1', 'question2']]
x_train = x_train.loc[:,cols].values
x_test = x_test.loc[:, cols].values
tfidf_w2v_q1_train, tfidf_w2v_q2_train = tfidf_w2v_q1_train.values, tfidf_w2v_q2_train.values
tfidf_w2v_q1_test, tfidf_w2v_q2_test = tfidf_w2v_q1_test.values, tfidf_w2v_q2_test.values
X_train = np.hstack((x_train, tfidf_w2v_q1_train, tfidf_w2v_q2_train))
X_test = np.hstack((x_test, tfidf_w2v_q1_test, tfidf_w2v_q2_test))
#let's split the X_train into train and CV datasets
X_train, X_cv , Y_train, Y_cv = train_test_split(X_train, y_train, stratify = y_train, test_size=0.2, random_state = 42)
std_data = StandardScaler()
std_data.fit(X_train)
X_train = std_data.transform(X_train) #Y_train
X_cv = std_data.transform(X_cv) #Y_cv
X_test = std_data.transform(X_test) #y_test


# for class_weight
a = Y_train.value_counts()
cl_weight = {0: round(a[1]/(a[0]+a[1]),2), 1: round(a[0]/(a[0]+a[1]), 2)}
cl_weight

#for TFIDF
if os.path.isfile('tfidf_GBDT_model.pkl'):
    with open('tfidf_GBDT_model.pkl', 'rb') as f:
        calib_gbdt_tfidf = pickle.load(f)

    #finding the log-loss
    y_prob_train = calib_gbdt_tfidf.predict_proba(tfidf_x_train)
    y_prob_test = calib_gbdt_tfidf.predict_proba(tfidf_x_test)
    print("\nLog loss on train Data with TFIDF vector using GBDT:",round(log_loss(Y_train, y_prob_train, eps=1e-15),4))
    print("Log loss on test Data with TFIDF vector using GBDT :",round(log_loss(y_test, y_prob_test, eps=1e-15),4))
else:

    print('Training the model:')
    XGB = LGBMClassifier(boosting_type = 'gbdt', max_depth = 10, n_estimators= 500,
                         class_weight = cl_weight, random_state=42)
    XGB.fit(tfidf_x_train, tfidf_y_train.values)
    calib_gbdt_tfidf = CalibratedClassifierCV(XGB, method="sigmoid")
    calib_gbdt_tfidf.fit(tfidf_x_train, tfidf_y_train.values)

    #finding the log-loss
    y_prob_train = calib_gbdt_tfidf.predict_proba(tfidf_x_train)
    y_prob_test = calib_gbdt_tfidf.predict_proba(tfidf_x_test)
    print("\nLog loss on train Data using Random Model",round(log_loss(Y_train, y_prob_train, eps=1e-15),4))
    print("Log loss on test Data using Random Model",round(log_loss(y_test, y_prob_test, eps=1e-15),4))

    #save the model
    with open('tfidf_GBDT_model.pkl', 'wb') as f:
        pickle.dump(calib_gbdt_tfidf, f)


#for TFIDF-w2v
if os.path.isfile('tfidf_w2v_GBDT_model.pkl'):
    with open('tfidf_w2v_GBDT_model.pkl', 'rb') as f:
        calib_gbdt_tfidf_w2v = pickle.load(f)
    #let's find out the log-loss
    y_prob_train = calib_gbdt_tfidf_w2v.predict_proba(X_train)
    y_prob_test = calib_gbdt_tfidf_w2v.predict_proba(X_test)
    print("Log loss on train Data with TFIDF W2V vector using GBDT:",round(log_loss(Y_train, y_prob_train, eps=1e-15),4))
    print("Log loss on test Data with TFIDF W2V vector using GBDT :",round(log_loss(y_test, y_prob_test, eps=1e-15),4))
else:
    # taking best hyper parameter: (why look into .ipynb file)
    XGB = LGBMClassifier(boosting_type = 'gbdt', max_depth = 8, n_estimators= 500,
                         class_weight = cl_weight, random_state=42)
    XGB.fit(X_train, Y_train.values)
    calib_gbdt_tfidf_w2v = CalibratedClassifierCV(XGB, method="sigmoid")
    calib_gbdt_tfidf_w2v.fit(X_train, Y_train.values)

    #let's find out the log-loss
    y_prob_train = calib_gbdt_tfidf_w2v.predict_proba(tfidf_x_train)
    y_prob_test = calib_gbdt_tfidf_w2v.predict_proba(tfidf_x_test)
    print("Log loss on train Data with TFIDF W2V vector using GBDT",round(log_loss(Y_train, y_prob_train, eps=1e-15),4))
    print("Log loss on test Data with TFIDF W2V vector using GBDT",round(log_loss(y_test, y_prob_test, eps=1e-15),4))

    #save the model
    with open('tfidf_w2v_GBDT_model.pkl', 'wb') as f:
        pickle.dump(calib_gbdt_tfidf_w2v, f)

#let's take input from users:
more_input = True
while more_input:
    new_df = pd.DataFrame(columns = ['question1','question2'])
    print('\nWrite first question:')
    q1 = input()
    print('Write seconde question:', )
    q2 = input()
    print('\n')

    new_df = new_df.append({'question1': q1, 'question2':q2}, ignore_index = True)
    new_df = extract_features(new_df) #getting advance and basic features

    # 'Getting the TFIDF-W2V feature'
    tfidf_w2v_q1_df = vectorize_question(new_df['question1'], word_to_idf_score)
    tfidf_w2v_q2_df = vectorize_question(new_df['question2'], word_to_idf_score)

    # Getting the TFIDF featues
    tfidf_X_q1, tfidf_X_q2 = tfidf1.transform(new_df['question1']), tfidf1.transform(new_df['question2'])

    # Creating final TFIDF W2V vector for given questions.......
    cols = [i for i in new_df.columns if i not in ['question1', 'question2']]
    new_df = new_df.loc[:,cols]
    new_df = new_df.values
    tfidf_w2v_q1 = tfidf_w2v_q1_df.values
    tfidf_w2v_q2 = tfidf_w2v_q2_df.values
    X_tfidf_w2v_q = np.hstack((new_df, tfidf_w2v_q1, tfidf_w2v_q2))
    X_tfidf_w2v_q = std_data.transform(X_tfidf_w2v_q)

    #Creating final TFIDF vector for given questions.......
    X_tfidf_q = hstack((tfidf_X_q1, tfidf_X_q2, new_df))

    print('\n\nPress 1 for TFIDF W2V\nPress 2 for TFIDF Only\nDeault will be TFIDF')
    try:
        pressed = int(input())
        if pressed == 1:
            pressed = 1
        else:
            pressed = 2
    except:
        pressed = 2

    if pressed == 1:
        y_q = calib_gbdt_tfidf_w2v.predict(X_tfidf_w2v_q)
        y_q_proba = calib_gbdt_tfidf_w2v.predict_proba(X_tfidf_w2v_q)
        print('\nPredicted class is: {}\nProbability of each class is {}'.format(y_q, y_q_proba))
    else:
        y_q = calib_gbdt_tfidf.predict(X_tfidf_q)
        y_q_proba = calib_gbdt_tfidf.predict_proba(X_tfidf_q)
        print('\nPredicted class is: {}\nProbability of each class is {}'.format(y_q, y_q_proba))

    print('\n')
    print("Do you want to check more: Pess 1 if yes\nOhterwise it'll terminate the session.")
    try:
        q3 = int(input())
        if q3 == 1:
            more_input = True
        else:
            more_input = False
    except:
        more_input = False
