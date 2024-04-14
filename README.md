# FastText Model Performance Compare to Traditional Techiques

In this repository we want to **compare** how **FastText** model **performance to** other **Models with different** **Preprocessing** and **Vecotrization** methods. In this work we are going to concentrate only on one task: **gender prediction**, it is related to high amount of time needed for preprocessing and fitting and testing the pipelines that we create, we tried lot's of ways to optimize, but here is the problem with models (they need sufficient amount of time for being trained). If this work is going to be interested for people, than we may continue with this work and as well check what is better for different purposes except gender prediction.

## Description of Data

For this project we used gender data, which is represented in two columns: post on reddit, female (0,1) - binary data basically. The wight of csv file was approximatelly 400Mb and after data cleaning represent 44000 different posts, the proportin of posts writted by males and females was equaly distributed (50-to-50 approximately), which leads to apprximately 22000 posts from males and 22000 posts from females.

## Models used

- **Logistic Regression model** from sklearn 
- **Random Forest Classifier** from sklearn
- **Support Vector Classifier** from sklearn
- And finaly **FastText model** from FastText, this model doesn't require any vectorization, and runs pretty fast

## Vectorization Methods Used 

- **Tf-Idf Vectorizern** provided by sklearn
- **Count Vecotorizer** as well provided by sklearn
- **Word 2 Vec Vectorizer** provided by gensim, the vocabulary of the Vectorizer is from Google news

## Preprocessing flows that we used 

- **Flow 0**: Without any preprocessing, just pure text tokenized by the pipeline (0 minutes)
- **Flow 1**: Remove URL → Remove hashtags → Remove usernames → Remove emojis (30 minutes)
- **Flow 2**: Lowercasing → Remove punctuation → Remove numbers → Remove extra white spaces (10 minutes)
- **Flow 3**: Remove stop words → Lowercasing → Replace negations by antonyms (2 minutes)
- **Flow 4**: Lemmatization of words only (1 minute)
- **Flow 5**: Stemming of words only (10 minutes)
- **Flow 6**: Expand words and slang (e.g., "lol" → "laughing out loud") (1 minute)
- **Flow 7**: Replace negation by antonyms (5 minutes)
- **Flow 8**: Multi-word grouping (not used due to high time complexity, reserved for future work) (6 hours)
- **Flow 9**: Remove URLs → Remove hashtags → Remove usernames → Remove emojis → Lemmatize words (30 minutes)
- **Flow 10**: Lowercasing → Remove punctuation → Remove numbers → Remove extra white spaces → Lemmatize words (10 minutes)
- **Flow 11**: Remove stop words → Replace negation by antonyms → Lemmatization of words (30 minutes)
- **Flow 12**: Remove URLs → Remove hashtags → Remove usernames → Remove emojis → Stemming of words (30 minutes)
- **Flow 13**: Lowercasing → Remove punctuation → Remove extra white spaces → Stemming of words (20 minutes)
- **Flow 14**: Remove stop words → Replace negations by antonyms → Stemming of words (4 minutes)
- **Flow 15**: Remove URLs → Remove hashtags → Remove usernames → Remove emojis → Word expansion (30 minutes)
- **Flow 16**: Lowercasing → Remove punctuation → Remove digits → Remove extra white spaces → Word expansion (10 minutes)
- **Flow 17**: Remove stop words → Replace negations by antonyms → Word expansion (30 minutes)
- **Flow 18** - remove urls -> remove hashtags -> remove usernames -> remove emojies -> multiword grouping (wasn't used in our case due to the high time complexity of method) (6 hours)
- **Flow 19** - lowercasing -> remove punctuation -> remove extra white spaces -> multi word grouping (wasn't used in our case due to the high time complexity of method) (6 hours)
- **Flow 20** - remove stop words -> replace negation by antonyms -> multiword grouping (wasn't used in our case due to the high time complexity of method)
## Description of files 

- **config.py** - in this file you will have to change the path to your data folder that we use.
- **functions_preprocessing.py** - this file consists of all functions that we used for preprocessing, there you can find detailed description of what different flows stand for and in which way we did preprocessing of the data.
- **functions_vectorization.py** - in this file you can observe all functions that we used for vectorization in our pipelines.
- **preprocess_dataframes.py** - this file was used for running different preprocessing flows on the data and saving in separate json file, so the pipelines in future suppose to work faster, and will not loose the progress of work that was done.
- **get_pipe_scores.py** - in this file we did the template script for creating the pipelines with different combination of preprocessing and vectorization and model, then we fit them and test in the same script and save step by step in the csv file.
- **results_extraction.py** - this file was used for extraction of the result from the csv files with results, we preprocess dataframes, taking out the best scores and combinations of preprocessing and vectorization.
- **train_fastext_model.py** - in this file you can find the script that we used to train and text the FastText model.
- **dataset_exploration.ipynb** - in this file we clean the initial data, this code repeats in **preprocessing_dataframse.py** as well.
- **debug_and_test.ipynb** - this file was used for experimenting, fising errors and debugging some code in above mentioned files. Don't look there, please;) (we don't want be responsible for your bleeding eyes).
- **presenting_results.ipynb** - in this file you can opbserve cool results tha we got after our looooooong work.
- **train_of_models.ipynb** - this file has more less exactly the same script as **get_pipe_scores.py**, but was used for different models, and this file was very handy to run the training and testing of the pipelines simultaniouly on one computer.
- **requirements.txt** - in this file you can observe all the libraries that we used for this project.

## How to set up enviroment 
1. Place all files that you downloaded in one folder
2. Create virtual envoroment (you can use conda but we prefer pip)
   
   MacOS and Linux
   ```
   python3 -m venv .venv
   ```
   Windows
   ```
   python -m venv .venv
   ```
3. Activate your enviroment
   
   MacOS and Linux
   ```
   source .venv/bin/activate
   ```
   Windows
   ```
   .venv\Scripts\activate
   ```
5. Install all libraries that we were using
   ```
   pip install -r requirements.txt
   ```
6. Before you are going to run the code you will have to specify path to the data folder in **config.py** file.
7. Thank you for attention to this repository, have fun while playing around with our code and experiments!:)
8. If you have problems with loading the wheel for FastText you probably need to download it manualy [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext) or [here](https://mirrors.aliyun.com/pypi/simple/fasttext-wheel/)
9. If you still have problems with running the code you can contact us or welcome into the StackOverFlow :))).

## Where to get the preprocessed data and initial data

1. Initial data can not be presented as we have obligation to not spread the dataset that we used, so it is impossible to get initial data :(
2. Preprocessed data: We are not sure if our obligation affects the preprocessed data as well, so we would like to keep this data away from the public.

## Where to load the models that we used
After tests we discovered that the models will load automaticaly so keep in mind that you may need enough space for these models. (approximatey 10GB)

