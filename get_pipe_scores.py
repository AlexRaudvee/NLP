### imports 
import pandas as pd
import gensim.downloader as api

from tqdm import tqdm
from fasttext import FastText
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from config import path_to_data_folder, path_to_fast_text_model

### LOAD THE MODELS
word2vec_model = api.load("word2vec-google-news-300") # model trained on lower case words, use lower case tokens
fast_model = FastText.load_model(path_to_fast_text_model)

# make neccesary imports for preprocessing and vectorizatio
from functions_vectorization import TfidfVectorizer, CountVectorizer, Word2VecVectorizer, FastTextVectorizer

list_of_preprocessed_data = ['/gender_df_preprocessed_0', '/gender_df_preprocessed_1', "/gender_df_preprocessed_2", '/gender_df_preprocessed_3', '/gender_df_preprocessed_4', '/gender_df_preprocessed_5', '/gender_df_preprocessed_6', '/gender_df_preprocessed_7', '/gender_df_preprocessed_9', '/gender_df_preprocessed_10', '/gender_df_preprocessed_11', '/gender_df_preprocessed_12', '/gender_df_preprocessed_13', '/gender_df_preprocessed_14', '/gender_df_preprocessed_15', '/gender_df_preprocessed_16', '/gender_df_preprocessed_17', '/gender_df_preprocessed_18']
list_of_vectorizers = [CountVectorizer]
list_of_models = [RandomForestClassifier]


# PIPELINES COBINATION AND IT'S SCORES FOR GENDER DATA
created_pipelines_scores = {}
for model in list_of_models:
    for vectorizer in list_of_vectorizers:
        for preprocessed_data in list_of_preprocessed_data:

            pipeline_name = f"pipeline_{preprocessed_data}_{vectorizer.__name__}_{model.__name__}"

            if vectorizer.__name__ == 'Word2VecVectorizer':
                pipeline = Pipeline([
                                ('vectorizer', vectorizer(word2vec_model)),
                                ('model', model())
                                ])
            elif vectorizer.__name__ == 'FastTextVectorizer':
                pipeline = Pipeline([
                                ('vectorizer', vectorizer(fast_model)),
                                ('model', model())
                                ])
            else:
                pipeline = Pipeline([
                                    ('vectorizer', vectorizer()),
                                    ('model', model())
                                    ])

            df = pd.read_json(f'{path_to_data_folder}{preprocessed_data}')

            X = df[f'{df.columns[0]}'].tolist()
            y = df[f'{df.columns[1]}'].tolist()

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            pipeline.fit(X_train, y_train)

            created_pipelines_scores[pipeline_name] = pipeline.score(X_test, y_test)

            tqdm_desc = f"{preprocessed_data}_{vectorizer.__name__}_{model.__name__}"
            tqdm.write(f"Finished: {tqdm_desc}")