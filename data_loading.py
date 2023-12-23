### imports
import pandas as pd

### data loading 
from config import data_path

# gender data loading
df_gender = pd.read_csv(f'{data_path}/gender.csv')

# extrovert introvert data loading
df_extrovert_introvert = pd.read_csv(f'{data_path}/extrovert_introvert.csv')

# feeling thinking data loading
df_feeling_thinking = pd.read_csv(f'{data_path}/feeling_thinking.csv')

# judging perceiving data loading
df_judging_perceiving = pd.read_csv(f'{data_path}/judging_perceiving.csv')

# nationality data loading 
df_nationality = pd.read_csv(f'{data_path}/nationality.csv')

# political leaning data loading 
df_political_leaning = pd.read_csv(f'{data_path}/political_leaning.csv')

# sensing intuitive data loading
df_sensing_intuitive = pd.read_csv(f'{data_path}/sensing_intuitive.csv')

# birth year data loading 
df_birth_year = pd.read_csv(f'{data_path}/birth_year.csv')