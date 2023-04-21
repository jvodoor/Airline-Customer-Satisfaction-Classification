import pandas as pd
import sys
#sys.path = ['c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src', 'c:\\Program Files\\Python311\\python311.zip', 'c:\\Program Files\\Python311\\DLLs', 'c:\\Program Files\\Python311\\Lib', 'c:\\Program Files\\Python311', '', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32\\lib', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\Pythonwin', 'c:\\Program Files\\Python311\\Lib\\site-packages', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\data', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\features', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\models']
import make_dataset as md


#categoricals: Gender, Customer Type, Type of Travel, Class, Inflight Distance - Cleanliness
#Gender, Customer Type, Type of Travel, Satisfcation (y)

#Function which creates ordinal encoded columns to be used on a categorical column. Each column will have a 1-z value representation,
#where z = the number of unique datapoints per column.
def df_encoding(df, column_name):
    list_uniques = df[column_name].unique().tolist()
    num_uniques = len(df[column_name].unique())
    dict_uniques = {}
    print(list_uniques)
    for i in range (num_uniques):
        dict_uniques[list_uniques[i]] = i
    column_name_ordinal = column_name + " Ordinal"
    df[column_name_ordinal]=df[column_name].map(dict_uniques)
    df = df.drop(column_name, axis=1)
    return df

def df_drop_many_cols(df, column_list):
    df = df.drop(columns = column_list)
    return df

def df_drop_0_values(df, column_list):
    for i in column_list:
        df = df[df[i] != 0]
    return df

#not quite sure yet how to adapt this so when people try the code for thesmelves they'll be able to open the dataset without needing to change the filepath.
df_feat_eng = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/external/comb df clean.csv")

#changing categoricals to numerical values
df_feat_eng = df_encoding(df_feat_eng, 'Gender') #0 = male, 1 = female
df_feat_eng = df_encoding(df_feat_eng, 'Type of Travel') #0 = personal, 1 = business
df_feat_eng = df_encoding(df_feat_eng, 'Class') #0 = eco plus, 1 = business, 2 = economy
df_feat_eng = df_encoding(df_feat_eng, 'satisfaction') #0 = not satisfied, 1 = satisfied
print(df_feat_eng.head(50))

drop_cols = ['Customer Type', 'Arrival Delay in Minutes', 'id'] #Customer Type being dropped due to
#vagueness of the column + badly spread data, Arrival Delay being dropped due to high correlation
#with departure delay in minutes, id being dropped because it is not something that is
#useful for modeling purposes

df_feat_eng = df_drop_many_cols(df_feat_eng, drop_cols)
print(df_feat_eng.info())

#removing 0 values from our possible datapoints as mentioned in our visualization section
drop_0_values = ['Inflight wifi service','Departure/Arrival time convenient', 
                 'Ease of Online booking', 'Gate location', 'Food and drink',
                   'Online boarding','Seat comfort', 'Inflight entertainment' ,
                   'On-board service', 'Leg room service','Baggage handling', 
                    'Checkin service', 'Inflight service', 'Cleanliness' ]
df_feat_eng = df_drop_0_values(df_feat_eng, drop_0_values)

print(df_feat_eng.describe())
print(df_feat_eng.info()) #reduced to 120k values, still quite high and sufficient for our purposes

print ((df_feat_eng['Departure Delay in Minutes'] != 0).value_counts())

df_feat_eng.to_csv("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/processed/df feat eng done.csv")