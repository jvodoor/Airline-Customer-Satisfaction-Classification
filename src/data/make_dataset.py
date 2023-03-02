import pandas as pd

def load_dataset(file_path):
    return pd.read_csv(file_path, index_col = 0)
def print_df_preliminary_contents(data):
    print(data.head())
    print(data.info(verbose = 0))
def merge_data (df1, df2):
    df_comb = pd.concat([df1, df2], axis = 0)
    return df_comb


#df_old_train = load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/external/train.csv")
#df_old_test = load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/external/test.csv")
#print_df_preliminary_contents(df_old_train)
#print_df_preliminary_contents(df_old_test)
#df = merge_data(df_old_train,df_old_test)
#print_df_preliminary_contents(df)
#df = df.dropna()
#print_df_preliminary_contents(df)
#df.to_csv("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/external/comb df clean.csv")