# quora-question-pairs
This project classifies whether given a one-pair of questions are duplicate or not.
There are total 6 files (excluding README.md). Two are with *.py* extension, two with *.ipynb* extension and two with *.pkl* extension

## *.ipynb* Extension
With this extension, files are IPython Notebooks. File name *Quora_EDA.ipynb* has exploratory data anaysis. In exploartory data analysis various distribution plots, barplots, t-SNE plots etc are plotted. In this file some additional features are designed. In *quora_vectorizing_and_models.ipynb* file first TF-IDF and TFIDF-W2V vectorization are done then final TF-IDF and TFIDF-W2V features are created by merging vectorized features with designed features in *Quora_EDA.ipynb* file.
Logistic regression, linear SVM and GBDT (with LightGBM) machine learning models are applied. For each model hyper-parameter tunning has been done. At the end all model's log-loss values are compared.

## *.py* Extension
*feature_extraction_functions.py* has important functions and *train_and_test.py* uses functions of *feature_extraction_functions.py* to create the features and vectorize the text-vector. Then best model and best parameter has been taken from *quora_vectorizing_and_models.ipynb* file and trained the model. 
This *train_and_test.py* also take input from users and for real-time testin.

## *.pkl* Extension
*tfidf_GBDT_model.pkl* and *tfidf_w2v_GBDT_model.pkl* are trained model with TFIDF and TFIDF-W2V vectorized features respectivelly. 
