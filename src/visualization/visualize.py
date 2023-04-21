import pandas as pd
import sys
#sys.path = ['c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src', 'c:\\Program Files\\Python311\\python311.zip', 'c:\\Program Files\\Python311\\DLLs', 'c:\\Program Files\\Python311\\Lib', 'c:\\Program Files\\Python311', '', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\win32\\lib', 'C:\\Users\\jvodo\\AppData\\Roaming\\Python\\Python311\\site-packages\\Pythonwin', 'c:\\Program Files\\Python311\\Lib\\site-packages', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\data', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\features', 'c:\\Users\\jvodo\\DATA 4950\\DATA-4950-Capstone\\src\\models']
import make_dataset as md
import matplotlib.pyplot as plt
import seaborn as sns

df_vis = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/external/comb df clean.csv")
md.print_df_preliminary_contents(df_vis)
x, y = md.x_y_split(df_vis, -1)


#plots for our satisfaction surveys, the ones where we have a 0-5 score

plt.figure(figsize = (16,8))
plt.subplot(131)
sns.countplot(data=df_vis, x='Inflight wifi service')
plt.subplot(132)
sns.countplot(data=df_vis, x='Departure/Arrival time convenient')
plt.subplot(133)
sns.countplot(data=df_vis, x='Ease of Online booking')
plt.show()
plt.close()

plt.figure(figsize = (16,8))
plt.subplot(131)
sns.countplot(data=df_vis, x='Gate location')
plt.subplot(132)
sns.countplot(data=df_vis, x='Food and drink')
plt.subplot(133)
sns.countplot(data=df_vis, x='Seat comfort')
plt.show()
plt.close()

plt.figure(figsize = (16,8))
plt.subplot(131)
sns.countplot(data=df_vis, x='Inflight entertainment')
plt.subplot(132)
sns.countplot(data=df_vis, x='On-board service')
plt.subplot(133)
sns.countplot(data=df_vis, x='Leg room service')
plt.show()
plt.close()

plt.figure(figsize = (16,8))
plt.subplot(131)
sns.countplot(data=df_vis, x='Baggage handling')
plt.subplot(132)
sns.countplot(data=df_vis, x='Checkin service')
plt.subplot(133)
sns.countplot(data=df_vis, x='Cleanliness')
plt.show()
plt.close()

plt.figure(figsize = (16,8))
plt.subplot(131)
sns.countplot(data=df_vis, x='Inflight service')
plt.show()
plt.close()

#there appears to be a fairly normal dispersion across the board, with some variation
#inbetween for all of our customer satisfaction scores. We've identified some 0s that are
#present in some scores but some that aren't in others. Due to the fact that 0's are fairly
#low count, we'll try to remove these in our feature engineering and determine if that's
#a good idea. Even if we removed say 10-20% of our values, we still are looking at 100,000
#total counts, plenty of data to build a consensus.


#Customer Type, Age, Type of Travel, Class, Flight Distance, Departure Delay in Minutes,
#Arrival delay in minutes, satisfaction

df_vis.describe()
plt.figure(figsize = (16,8))
plt.subplot(131)
sns.countplot(data=df_vis, x='Gender')
plt.subplot(132)
sns.countplot(data=df_vis, x='Customer Type')
plt.subplot(133)
sns.countplot(data=df_vis, x='Type of Travel')
plt.show()
plt.close()


plt.figure(figsize = (16,8))
plt.subplot(131)
sns.countplot(data=df_vis, x='Class')
plt.subplot(132)
sns.countplot(data=y, x='satisfaction')
plt.show()
plt.close()

#We seem to be skewing towards business travelers with our customer counts, though
#there are a fair amount of economy passengers. Gender is fairly even, while most of our
#customers would be considered "loyal customers". I may exclude loya/disloyal because
#it feels too vague to be useful. With eco plus' low counts, I may opt to combine it
#with eco, or just let it go as its own and see what happens.


#heat map of numerical values
fig = plt.figure(figsize= (8,6))
sns.heatmap(df_vis[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']].corr(), cmap="viridis", annot=True)
plt.show()
#looking at our correlated numerical variables, unsurprisingly there's a strong correlation
#between Departure and Arrival Delays. Might be worth exploring those 2 some more and decide
#to keep 1 or the other. 


#exploring Arrival and Departure delay in more detail
print(df_vis['Departure Delay in Minutes' ].describe())
print(df_vis['Arrival Delay in Minutes'].describe())
print("")
print("non zero Departure delays", df_vis['Departure Delay in Minutes'].astype(bool).sum(axis=0))
print("")
print("non zero arrival delays", df_vis['Arrival Delay in Minutes'].astype(bool).sum(axis=0))

#the numbers are very similar to each other. May try 2 options where I keep both in or remove 1 or both possibly











# This is exploring how to make a sequence of scrollable graphs, doesn't work yet, need
#to do more research.
'''
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# create a list of graph objects
graphs = []
for i in range(5):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [i+1, i+2, i+3])
    graphs.append(fig)

# create a figure to hold the navigation buttons
fig_nav = plt.figure()
ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
btn_prev = Button(ax_prev, 'Prev')
btn_next = Button(ax_next, 'Next')

# define the callback functions for the navigation buttons
def prev(event):
    plt.close(graphs[i])
    i = (i - 1) % len(graphs)
    graphs[i].show()

def next(event):
    plt.close(graphs[i])
    i = (i + 1) % len(graphs)
    graphs[i].show()

# connect the navigation buttons to their callback functions
btn_prev.on_clicked(prev)
btn_next.on_clicked(next)

# show the first graph in the sequence
i = 0
graphs[i].show()'''