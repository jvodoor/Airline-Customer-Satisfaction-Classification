import pandas as pd
import sys
sys.path.insert(0, '/Users/jvodo/DATA 4950/DATA-4950-Capstone/src/data')
import make_dataset as md

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

#categoricals: Gender, Customer Type, Type of Travel, Class, Inflight Distance - Cleanliness
#Gender, Customer Type, Type of Travel, Satisfcation (y) = binary encoding
'''df_feat_eng = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/external/comb df clean.csv")
#md.print_df_preliminary_contents(df_feat_eng)
x, y = md.x_y_split(df_feat_eng, -1)
#categoricals: Gender, Customer Type, Type of Travel, Class, Inflight Distance - Cleanliness
#Gender, Customer Type, Type of Travel, Satisfcation (y) = binary encoding
# Class = ordinal number encoding
df_feat_eng = df_encoding(df_feat_eng, "Gender")
print(df_feat_eng.head())
print(df_feat_eng.info())'''