import pandas as pd
import sys
sys.path.insert(0, '/Users/jvodo/DATA 4950/DATA-4950-Capstone/src/data/')
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


#not quite sure yet how to adapt this so when people try the code for thesmelves they'll be able to open the dataset without needing to change the filepath.
df_feat_eng = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/external/comb df clean.csv")

x, y = md.x_y_split(df_feat_eng, -1)
df_feat_eng = df_encoding(df_feat_eng, "Gender")
print(df_feat_eng.head())
print(df_feat_eng.info())



#to be done: finish encoding our categoricals, eliminate Customer Type (disloyal/loyal), eliminate either departure or arrival delay time, potentially remove all 0 values from our survey data.
#evaluate if this doesn't dilute our datapoints too much
