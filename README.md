# Development of fasttext model performance

In this project we want to focus on how to improve performance of fasttext model after various type of preprocessing, fine tunning and hyperparameter tunning.

## Description of files

- **config.py** - in this file you will have to change the path to your data folder that we use
- **data_loading.py** - in this file we are doing data loading of all dataframes that are avaliable
- **exploration.ipynb** - in this jypiter notebook file we are doing first and basic analysis of the data that we have, and choose the data that we think is going to be valuable for our purpose
- **functions.py** - in this file we are writing basic functions that we are going to use for preprocessing and for getting the results (train the model, tune the model, and bla bla bla)
- **main.py** main file that we are going to use as final stage when applying model on the data and getting it performance (would be removed in future)
- **preprocessing.py** - in this file we are doing preprocessing of the text from the dataframe that we choosed
- **requirements.txt** - in this file you can observe all the libraries that we used for this project

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
