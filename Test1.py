import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))


df = pd.read_csv("games.csv")
print_tabulate(df)
winner = df['winner']
winner1 = 0
winner2 = 0
for x in winner:
    if x == 1:
        winner1 = winner1 + 1
    else:
        winner2 = winner2 + 1

firstBlood = df['firstBlood']
firstBlood1 = 0
firstBlood2 = 0
for x in firstBlood:
    if x == 1:
        firstBlood1 = firstBlood1 + 1
    else:
        firstBlood2 = firstBlood2 + 1

firstTower = df['firstTower']
firstTower1 = 0
firstTower2 = 0
for x in firstTower:
    if x == 1:
        firstTower1 = firstTower1 + 1
    else:
        firstTower2 = firstTower2 + 1

firstInhibitor = df['firstInhibitor']
firstInhibitor1 = 0
firstInhibitor2 = 0
for x in firstInhibitor:
    if x == 1:
        firstInhibitor1 = firstInhibitor1 + 1
    else:
        firstInhibitor2 = firstInhibitor2 + 1

firstBaron = df['firstBaron']
firstBaron1 = 0
firstBaron2 = 0
for x in firstBaron:
    if x == 1:
        firstBaron1 = firstBaron1 + 1
    else:
        firstBaron2 = firstBaron2 + 1

firstDragon = df['firstDragon']
firstDragon1 = 0
firstDragon2 = 0
for x in firstDragon:
    if x == 1:
        firstDragon1 = firstDragon1 + 1
    else:
        firstDragon2 = firstDragon2 + 1

firstRiftHerald = df['firstRiftHerald']
firstRiftHerald1 = 0
firstRiftHerald2 = 0
for x in firstRiftHerald:
    if x == 1:
        firstRiftHerald1 = firstRiftHerald1 + 1
    else:
        firstRiftHerald2 = firstRiftHerald2 + 1

labels = ['Winner', 'FirstBlood', 'FirstTower', 'FirstInhibitor', 'FirstBaron', 'FirstDragon', 'FirstRiftHerald']
Team1_means = [winner1, firstBlood1, firstTower1, firstInhibitor1, firstBaron1, firstDragon1, firstRiftHerald1]
Team2_means = [winner2, firstBlood2, firstTower2, firstInhibitor2, firstBaron2, firstDragon2, firstRiftHerald2]
width = 0.55  # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, Team1_means, width, label='Team 1')
ax.bar(labels, Team2_means, width, bottom=Team1_means, label='Team 2')

ax.set_ylabel('Score')
ax.set_title('Scores Team1 vs Team2')
ax.legend()

plt.show()
