import sys
sys.path.insert(0, '/Users/jvodo/DATA 4950/DATA-4950-Capstone/src/data/')
sys.path.insert(0, '/Users/jvodo/DATA 4950/DATA-4950-Capstone/src/features/')
import make_dataset as md
import build_features as bf
import pandas as pd

df_train = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/external/comb df clean.csv")
df_train = bf.df_encoding(df_train, 'Gender') #0 = male, 1 = female
df_train = bf.df_encoding(df_train, 'Type of Travel') #0 = personal, 1 = business
df_train = bf.df_encoding(df_train, 'Class') #0 = eco plus, 1 = business, 2 = economy
df_train = bf.df_encoding(df_train, 'satisfaction') #0 = not satisfied, 1 = satisfied
print(df_train.head(50))

drop_cols = ['Customer Type', 'Arrival Delay in Minutes'] #Customer Type being dropped due to
#vagueness of the column + badly spread data, Arrival Delay being dropped due to high correlation
#with departure delay in minutes

df_train = bf.df_drop_many_cols(df_train, drop_cols)
print(df_train.info())

#removing 0 values from our possible datapoints as mentioned in our visualization section
drop_0_values = ['Inflight wifi service','Departure/Arrival time convenient', 
                 'Ease of Online booking', 'Gate location', 'Food and drink',
                   'Online boarding','Seat comfort', 'Inflight entertainment' ,
                   'On-board service', 'Leg room service','Baggage handling', 
                    'Checkin service', 'Inflight service', 'Cleanliness' ]
df_train = bf.df_drop_0_values(df_train, drop_0_values)

print(df_train.describe())
print(df_train.info()) #reduced to 120k values, still quite high and sufficient for our purposes


#X, Y = md.x_y_split(df_train, -1)


