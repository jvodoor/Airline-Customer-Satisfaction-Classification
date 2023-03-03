import pandas as pd
import sys
sys.path.insert(0, '/Users/jvodo/DATA 4950/DATA-4950-Capstone/src/data')
import make_dataset as md
import matplotlib.pyplot as plt
import seaborn as sns

df_vis = md.load_dataset("C:/Users/jvodo/DATA 4950/DATA-4950-Capstone/data/external/comb df clean.csv")
md.print_df_preliminary_contents(df_vis)
x, y = md.x_y_split(df_vis, -1)

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
















# This is exploring how to make a sequence of scrollable graphs, needs more exploration
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