import pandas as pd
import plotly.express as px

from config import path_to_data_folder
from sklearn.utils import shuffle
from tqdm import tqdm

from functions_preprocessing import flow_preprocessing_1, flow_preprocessing_2, flow_preprocessing_3, flow_preprocessing_4, flow_preprocessing_5, flow_preprocessing_6, flow_preprocessing_7, flow_preprocessing_8, flow_preprocessing_9, flow_preprocessing_10, flow_preprocessing_11, flow_preprocessing_12, flow_preprocessing_13, flow_preprocessing_14, flow_preprocessing_15, flow_preprocessing_16, flow_preprocessing_17, flow_preprocessing_18, flow_preprocessing_19, flow_preprocessing_20, flow_preprocessing_21

tqdm.pandas()


def num_of_words(df: pd.DataFrame)-> int:
    '''
    :param df: a data frame with sentences in column "post"
    :type df: a data frame
    :returns: the number of words in the dataset
    :rtype: int
    '''
    count = 0
    iterator = df['post']
    for i in iterator:
        count += len(i.split())
    return count

def num_of_cars(df: pd.DataFrame)-> int:
    '''
    :param df: a data frame with sentences in column "post"
    :type df: a data frame
    :returns: the number of carachters in the dataset
    :rtype: int
    '''
    count = 0
    iterator = df['post']
    for i in iterator:
        count+= len([j for j in i])
    return count

def avg_num_of_words(df: pd.DataFrame)-> int:
    '''
    :param df: a data frame with sentences in column "post"
    :type df: a data frame
    :returns: the average number of words in the dataset
    :rtype: int
    '''
    return round(num_of_words(df)/len(df))

def avg_num_of_cars(df: pd.DataFrame) -> int:
    '''
    :param df: a data frame with sentences in column "post"
    :type df: a data frame
    :returns: the average number of carachters in the dataset
    :rtype: int
    '''
    return round(num_of_cars(df)/len(df))

def num_of_cat(df: pd.DataFrame) -> int:
    '''
    :param df: a data frame
    :type df: a data frame
    :returns: the number of categories in the dataset
    :rtype: int
    '''
    last_col = df.columns[-1]
    return df[last_col].nunique()

def missing_values(df: pd.DataFrame) -> str:
    '''
    :param df: a data frame
    :type df: a data frame
    :returns: if there are any missing values in the dataset
    :rtype: str
    '''
    count = df.isnull().sum().sum()
    if count == 0:
        return print(f'There are no missing values!')
    return print(f'There are missing values!')

def return_vis(df: pd.DataFrame, x_label: str, y_label: str, title: str) -> px.bar:
    '''
    :param df: a data frame with target and feature columns
    :type df: a data frame
    :param x_label: name for the x-axis
    :type x_label: str
    :param y_label: name for the y-axis
    :type y_label: str
    :param title: title describing the barplot
    :type title: str
    :returns: a barplot with percentage distribution of classes in the dataset
    :rtype: px.bar
    '''
    lst = df[df.columns[-1]].unique()
    new_lst = [df[df[df.columns[-1]] == value].count()['post']/ len(df) * 100 for value in lst]
    df1 = pd.DataFrame(data = {'count': new_lst, df.columns[-1]: lst})

    fig = px.bar(df1, x = df.columns[-1], y = 'count', 
                    labels={
                            df.columns[-1]: x_label,
                            "count": y_label,
                            "species": lst
                            }
                    )
    fig.update_layout(title_text = title)
    fig.show();
    return 

# Reading datasets
gender_df = pd.read_csv(f'{path_to_data_folder}/gender.csv')

jud_per_df = pd.read_csv(f'{path_to_data_folder}/judging_perceiving.csv')

political_df  = pd.read_csv(f'{path_to_data_folder}/political_leaning.csv')


# slice data for test run
# gender_df = gender_df[:100]
# political_df = political_df[:100]
# jud_per_df = jud_per_df[:100]


# Removing irrelevant columns
gender_df = gender_df.drop('auhtor_ID', axis = 1)
political_df = political_df.drop('auhtor_ID', axis = 1)
jud_per_df = jud_per_df.drop('auhtor_ID', axis = 1)

# Removing duplicates
gender_df = gender_df[gender_df.duplicated() == False]
political_df = political_df[political_df.duplicated() == False]
jud_per_df = jud_per_df[jud_per_df.duplicated() == False]

# Undersampling political_df
minority_class = political_df[political_df['political_leaning'] == 'left']
majority_class1 = political_df[political_df['political_leaning'] == 'center']
majority_class2 = political_df[political_df['political_leaning'] == 'right']

majority_class1 = shuffle(majority_class1)[:len(minority_class)]
majority_class2 = shuffle(majority_class2)[:len(minority_class)]

political_df = shuffle(pd.concat([minority_class, majority_class1, majority_class2])).reset_index(drop = True)

# Undersampling judging percieving dataset
minority_class = jud_per_df[jud_per_df['judging'] == 0]
majority_class = jud_per_df[jud_per_df['judging'] == 1]

majority_class = shuffle(majority_class)[:len(minority_class)]

jud_per_df = shuffle(pd.concat([minority_class, majority_class])).reset_index(drop = True)

# preprocess text and safe in file

gender_df['post'] = gender_df['post'].progress_apply(lambda text: flow_preprocessing_4(text))
gender_df.to_json('gender_df_preprocessed_4')