### imports 
import csv

import pandas as pd
import gensim.downloader as api

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from config import path_to_data_folder

### LOAD THE MODELS
word2vec_model = api.load("word2vec-google-news-300") # model trained on lower case words, use lower case tokens

# make neccesary imports for preprocessing and vectorizatio
from functions_vectorization import TfidfVectorizer, CountVectorizer, Word2VecVectorizer, FastTextVectorizer

list_of_preprocessed_data = ['gender_df_preprocessed_0', 'gender_df_preprocessed_1', "gender_df_preprocessed_2", 'gender_df_preprocessed_3', 'gender_df_preprocessed_4', 'gender_df_preprocessed_5', 'gender_df_preprocessed_6', 'gender_df_preprocessed_7', 'gender_df_preprocessed_9', 'gender_df_preprocessed_10', 'gender_df_preprocessed_11', 'gender_df_preprocessed_12', 'gender_df_preprocessed_13', 'gender_df_preprocessed_14', 'gender_df_preprocessed_15', 'gender_df_preprocessed_16', 'gender_df_preprocessed_17']
list_of_vectorizers = [TfidfVectorizer]
list_of_models = [RandomForestClassifier, LogisticRegression]


# PIPELINES COBINATION AND IT'S SCORES FOR GENDER DATA
df = pd.read_csv('scores.csv', header=None, names=['pipeline_name', 'pipeline_scores'])

# Extract the pipeline names
finished_ = df['pipeline_name'].tolist()

file_path = "scores.csv"
for model in list_of_models:
    for vectorizer in list_of_vectorizers:
        for preprocessed_data in list_of_preprocessed_data:

            pipeline_name = f"pipeline_{preprocessed_data}_{vectorizer.__name__}_{model.__name__}_2N"

            if pipeline_name not in finished_:
                pipeline = Pipeline([
                                    ('vectorizer', vectorizer()),
                                    ('model', model())
                                    ])

                df = pd.read_json(f'{path_to_data_folder}/{preprocessed_data}.json')

                X = df[f'{df.columns[0]}'].tolist()
                y = df[f'{df.columns[1]}'].tolist()

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                # fi the pipe
                pipeline.fit(X_train, y_train)

                # Predict on the test set
                y_pred = pipeline.predict(X_test)

                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

                input_in_the_file = [pipeline_name, [f'Score: {pipeline.score(X_test, y_test)}', f'precision: {precision_score(y_test, y_pred)}', f'Recall: {recall_score(y_test, y_pred)}', f'ROC AUC: {roc_auc_score(y_test, y_pred_proba)}']]

                # Append new data to the CSV file
                with open(file_path, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(input_in_the_file)

                print(f'{preprocessed_data} {vectorizer.__name__} {model.__name__} N2 finished and stored')

                finished_.append(pipeline_name)
            else: 
                continue