import os
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import pandas as pd

##################################################### IMPORTANT ########################################################################
# Before running the script make sure that:
# 1. variable "folder_path" (line 58) is defined, path should contain folders of prerpocessed data in JSON format.
#    If training on different data sets, last charackter in the name of the file should be an unique number, in following format:
#    "data_0", "data_preprocessed_10". EACH DATA FILE SHOULD END WITH A UNIQUE NUMBER, PREJUDICED WITH UNDERSCORE!
#
# 2. variable "file_path_train" (line 91) is defined, create a folder named "data_preprocessed_text", there all text files required for
#    model training are going to be stored. These text files will be later called in (line 105).
#
# 3. variabe "inside_folder"  (line 98) is defined, create an unique folder that stores trained model on given data sets
#
# 4. name of the csv file with model performance is created, otherwise models'results will not be saved in a csv format
# 
# 5. In line 105 you can specify hyperparametrs when training the model, additional examples are commented
#######################################################################################################################################
 


def preparing_data_fasttext(file_path: str, df: pd.DataFrame, col1: str, col0: str):
        """This function gets as an input file path to new text file, which will be created from a data frame, while using 
            tareget and predictor labels from the data set.

            Parameters: 
            file_path: str   -> it is a string which specifies where a text file with transformed data is going to be created. Make sure what 
                                is activited.
            df: pd.DataFrame -> this data frame contains info from trainig or test data frame
            col0: str        -> choses a column form data frame resposible for the text data
            col1: str        -> choses a column from data frame responsible for the label data
        """
        with open(file_path, 'w',encoding='utf-8') as file:
            for index, row in df.iterrows():
                text = f'__label__{row[col1]} {row[col0]}'
                file.write(text + '\n')

####################################### COLUMN DESCRIPTION OF EVALUATION DATA FRAME ###################################################
# Specifies the name of columns in in data frame with main statistics, 
# Preprocessing flow: indexes how given data set was preprocessed (different preprocessing methods applied in differnrt combinations)
# Precison: Evaluates the precision of the model
# Recall: Evaluete the recall of the model
# F1: Evaluates the F1 statistic of the model
# ROC_AUC: Evaluates the ROC_AUC of the model
#######################################################################################################################################
                
columns: list[str] = ['Preprocessing flow', 'Accuracy', 'Precision', 'Recall', "F1", 'ROC_AUC']

# Data frame describing every trained model
stats_df = pd.DataFrame(columns=columns)

# path were preprocessed files are being stored
folder_path = r'data_preprocessed_df'

# List all files in the folder
files = os.listdir(folder_path)

# number of files in the folder with preprocessed data
count_files = len(files)

# Iterate through the preprocessed files
for file_index in range(count_files):

    # gets the file path of the file
    file_path = os.path.join(folder_path, files[file_index])
    df = pd.read_json(file_path)

    file_name = os.path.basename(file_path)
    num_of_flow = os.path.splitext(file_name)[0].split("_")[-1]

    # used for reproducability of the code (sets the seed for the splitting the data)
    seed_value = 0

    # getting the data frames with target and predictors
    X = df[df.columns[0]]
    y = df[df.columns[1]]

    # splitting of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

    # getting the data frames to be converted to text files
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # path to text file with training data
    file_path_train = rf"data_preprocessed_text\text_train_{num_of_flow}.txt"

    # this block checks if there exists the text file needed for training the model, made for optimisation
    if not os.path.isfile(file_path_train):
        preparing_data_fasttext(file_path_train, df_train, df.columns[1], df.columns[0])

    # specifien in what folder with models newly modeles will be saved
    inside_folder = "\\model_raw"

    # path where the model will be saved
    save_path = f'fasttext_models{inside_folder}\model{num_of_flow}.bin'

    # chcks if the model exists in a folders with given model types
    if not os.path.isfile(save_path):
        model = fasttext.train_supervised(input=rf"data_preprocessed_text\text_train_{num_of_flow}.txt") #, lr=1.0, epoch=25, # trains the model
        model.save_model(save_path)
    else:
        model =  fasttext.load_model(save_path)

    # predicts labels for the test data
    y_predicted = [int(model.predict(text)[0][0][-1])for text in X_test]

    # computes precision, recall fscore and support of the model (support not included in final evaluation)
    precision, recall, fscore, support= score(y_test, y_predicted)
    
    # computes the accuracy of the model
    accuracy = accuracy_score(y_test, y_predicted)
    
    # gets the false positive rates and true positive rates and treshholds
    fpr, tpr, thresholds = roc_curve(y_test, y_predicted)

    # gets the roc_auc of the model
    roc_auc = auc(fpr, tpr)

    # creates the tuples with data of every model, one model per row
    row_dict = dict(zip(columns, (f'processing_flow_{num_of_flow}', accuracy, precision[1], recall[1], fscore[1], roc_auc)))
    row_df = pd.DataFrame([row_dict], columns=columns)
    stats_df = pd.concat([stats_df, row_df], ignore_index=True)

    print(f'{file_index}') # keeps track of models are being created


stats_df.to_csv("results/fasttext_results_raw_model.csv") # convert result data frame to csv for later anaysis