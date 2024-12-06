#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 3 17:19:19 2024

@author: williamsempire
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

#Set the working directory
the_path = "/Users/williamsempire/Desktop/Columbia/Courses/5067 NLP/data"

#Construct the path to file
file_path = os.path.join(the_path, "hw4.pk")

#Load the dataset
with open(file_path, 'rb') as f:
    data = pickle.load(f)

#Examine the data structure
print(data.sample(n=20))

#Create df and text type
df = pd.DataFrame(data, columns=['text', 'label'])
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(str)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

#Build the Pipeline with TF-IDF and Classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('classifier', MultinomialNB())
])

#Model Training
pipeline.fit(X_train, y_train)

#Model Evaluation
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Save the pipeline for future use
import joblib
joblib.dump(pipeline, 'document_classifier.pkl')

#Predicting new data
def classify_new_documents(texts):
    model = joblib.load('document_classifier.pkl')
    predictions = model.predict(texts)
    return predictions

#Example prediction
new_docs = ["This is a legal contract regarding compliance.", 
            "Our new marketing campaign targets social media.",
            "The code documentation details API usage."]
predictions = classify_new_documents(new_docs)

for doc, label in zip(new_docs, predictions):
    print(f"Document: {doc}\nPredicted Label: {label}\n")



