### data loading 
from data_loading import df_birth_year, df_extrovert_introvert, df_feeling_thinking, df_gender, df_judging_perceiving, df_nationality, df_political_leaning, df_sensing_intuitive

######### see the destribution of the data #########

print('df_gender describtion')
print(df_gender.describe())

print('df_extrovert_introvert describtion')
print(df_extrovert_introvert.describe())

print('df_feeling_thinking describtion')
print(df_feeling_thinking.describe())

print('df_judging_perceiving describtion')
print(df_judging_perceiving.describe())

print('df_nationality describtion')
print(df_nationality.describe())

print('df_political_leaning describtion')
print(df_political_leaning.describe())

print('df_sensing_intuitive describtion')
print(df_sensing_intuitive.describe())

print('df_birth_year describtion')
print(df_birth_year.describe())