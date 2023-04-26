import pandas as pd 
import numpy as np 
import os

#function which will load initial dataset
def load_dataset(file_path):
    return pd.read_csv(file_path, index_col = 0)


#printing basic exploratory info for dataset, mainly to make sure the dataset loaded and prints correct ouput
def print_df_preliminary_contents(data):
    print(data.head())
    print(data.info(verbose = 0))

#The dataset came as 2 sepaarte files, so they are concacatenated together to form a combined CSV
def merge_data (df1, df2):
    df_comb = pd.concat([df1, df2], axis = 0)
    return df_comb

#creating x df and y df, where y represents our target variable nad x represents every column but the target
#this has customizability such that if we want to explore different columns as a target value, we can as long as we know
#the axis

def x_y_split (df1, column_axis):
    while column_axis < -1 or column_axis >= len(df1.axes[1]):
        print ("Invalid index, try again.")
        column_axis = int(input("Enter your axis you wish to split: "))
    if column_axis != -1:
        y = df1.iloc[:,column_axis:column_axis+1]
    else:
        y = df1.iloc[:,-1:]
    x = df1.drop(df1.iloc[:,-1:], axis = 1)
    return x,y


#opening file paths for users to load in datasets. File path points to default home directory of project folder.
#Home directory = "Data-4950-Capstone" project folder. If your datasets are in another folder, you must modify this
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '..', '..')


#load datasets
df_old_train = load_dataset(file_path + "/data/external/train.csv")
df_old_test = load_dataset(file_path + "/data/external/test.csv")

#merge data
df = merge_data(df_old_train,df_old_test)


#dropped nulls for our first pass of a cleaned up data set
df = df.dropna()

#save as dataset for feature engineering
df.to_csv(file_path + "/data/external/comb df clean.csv")
