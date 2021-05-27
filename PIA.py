import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))


# 2) Data Adquisition and Cleaning

df = pd.read_csv("games.csv")
df2 = pd.read_csv("games.csv")
df['creationTime'] = pd.to_datetime(df['creationTime'])


# 3) Data Parsing


def droper(df):
    dropcolumns = {'t1_champ1id', 't1_champ1_sum1', 't1_champ1_sum2', 't1_champ2id', 't1_champ2_sum1', 't1_champ2_sum2',
                   't1_champ3id', 't1_champ3_sum1', 't1_champ3_sum2', 't1_champ4id', 't1_champ4_sum1', 't1_champ4_sum2',
                   't1_champ5id', 't1_champ5_sum1', 't1_champ5_sum2', 't1_towerKills', 't1_inhibitorKills',
                   't1_baronKills',
                   't1_dragonKills', 't1_riftHeraldKills', 't1_ban1', 't1_ban2', 't1_ban3', 't1_ban4', 't1_ban5',
                   't2_champ1id',
                   't2_champ1_sum1', 't2_champ1_sum2', 't2_champ2id', 't2_champ2_sum1', 't2_champ2_sum2', 't2_champ3id',
                   't2_champ3_sum1', 't2_champ3_sum2', 't2_champ4id', 't2_champ4_sum1', 't2_champ4_sum2', 't2_champ5id',
                   't2_champ5_sum1', 't2_champ5_sum2', 't2_towerKills', 't2_inhibitorKills', 't2_baronKills',
                   't2_dragonKills',
                   't2_riftHeraldKills', 't2_ban1', 't2_ban2', 't2_ban3', 't2_ban4', 't2_ban5'
                   }
    for i in dropcolumns:
        df.drop(i, inplace=True, axis=1)


# 4) Decriptive Statistics

print('La partida con mayor duracion en minutos fue de: (MAX)')
print("{:.2f}".format(df['gameDuration'].max(axis=0) / 60))
print('La partida con menor duracion en minutos fue de: (MIN)')
print("{:.2f}".format(df['gameDuration'].min(axis=0) / 60))
print('El promedio de una partida es de : (MEAN)')
print("{:.2f}".format(df['gameDuration'].mean(axis=0) / 60))

print_tabulate(df.describe())

conditioner1 = df.loc[df['winner'] == 1]
conditioner1.drop('gameId', inplace=True, axis=1)
conditioner1.drop('creationTime', inplace=True, axis=1)
conditioner1.drop('gameDuration', inplace=True, axis=1)
conditioner1.drop('seasonId', inplace=True, axis=1)
print('Moda cuando cuando el quipo 1 gana')
print_tabulate(conditioner1.mode(dropna=False))

conditioner2 = df.loc[df['winner'] == 2]
conditioner2.drop('gameId', inplace=True, axis=1)
conditioner2.drop('creationTime', inplace=True, axis=1)
conditioner2.drop('gameDuration', inplace=True, axis=1)
conditioner2.drop('seasonId', inplace=True, axis=1)
print('Moda cuando cuando el quipo 2 gana')
print_tabulate(conditioner2.mode(dropna=False))

droper(df2)
df2.hist(bins=3)

# 5) Data Visualization

boxplot = df.boxplot(by='winner', column=['t1_towerKills'], grid=False)
boxplot.plot()
plt.show()

# 6) Statistic Test
winner = df['winner']
firstBlood = df['firstBlood']
firstTower = df['firstTower']
firstInhibitor = df['firstInhibitor']
firstBaron = df['firstBaron']
firstDragon = df['firstDragon']
firstRiftHerald = df['firstRiftHerald']
creationtime = df['creationTime']
gameDuration = df['gameDuration']

winner1 = 0
winner2 = 0
for x in winner:
    if x == 1:
        winner1 = winner1 + 1
    else:
        winner2 = winner2 + 1

firstBlood1 = 0
firstBlood2 = 0
for x in firstBlood:
    if x == 1:
        firstBlood1 = firstBlood1 + 1
    else:
        firstBlood2 = firstBlood2 + 1

firstTower1 = 0
firstTower2 = 0
for x in firstTower:
    if x == 1:
        firstTower1 = firstTower1 + 1
    else:
        firstTower2 = firstTower2 + 1

firstInhibitor1 = 0
firstInhibitor2 = 0
for x in firstInhibitor:
    if x == 1:
        firstInhibitor1 = firstInhibitor1 + 1
    else:
        firstInhibitor2 = firstInhibitor2 + 1

firstBaron1 = 0
firstBaron2 = 0
for x in firstBaron:
    if x == 1:
        firstBaron1 = firstBaron1 + 1
    else:
        firstBaron2 = firstBaron2 + 1

firstDragon1 = 0
firstDragon2 = 0
for x in firstDragon:
    if x == 1:
        firstDragon1 = firstDragon1 + 1
    else:
        firstDragon2 = firstDragon2 + 1

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
width = 0.55
fig, ax = plt.subplots()
ax.bar(labels, Team1_means, width, label='Team 1')
ax.bar(labels, Team2_means, width, bottom=Team1_means, label='Team 2')
ax.set_ylabel('Score')
ax.set_title('Scores Team1 vs Team2')
ax.legend()
plt.show()

# 7) Linear Models

n = 0

for x in gameDuration:
    n = n + 1


# Generador de distribución de datos para regresión lineal simple
def generador_datos_simple(beta, muestras, desviacion):
    # Genero n (muestras) valores de x aleatorios entre 0 y 100
    x = np.random.random(muestras) * 10
    # Genero un error aleatorio gaussiano con desviación típica (desviacion)
    e = np.random.randn(muestras) * desviacion
    # Obtengo el y real como x*beta + error
    y = x * beta + e
    return x.reshape((muestras, 1)), y.reshape((muestras, 1))


# Parámetros de la distribución
desviacion = df['t2_towerKills'].mean()
beta = 10

x, y = generador_datos_simple(beta, n, desviacion)

# Represento los datos generados
plt.scatter(x, y)

# Creo un modelo de regresión lineal
modelo = linear_model.LinearRegression()

# Entreno el modelo con los datos (X,Y)
modelo.fit(x, y)
# Podemos predecir usando el modelo
y_pred = modelo.predict(x)

# Representamos el ajuste (rojo) y la recta Y = beta*x (verde)
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
x_real = np.array([0, 10])
y_real = x_real * beta
plt.plot(x_real, y_real, color='green')
plt.show()

# 8) Forecasting

y = pd.to_datetime(df['creationTime'])
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y, color='red', marker='*', linestyle='-', linewidth=0.5)
ax.set_ylabel('creationTime')
ax.legend()
plt.xlim([0, 100])
plt.show()

# 9) Data classification

print(df.shape)
print(df.info(verbose=True))
print(df.groupby('winner').size())

fig, ax = plt.subplots()
plt.plot(df['t1_towerKills'], df['t2_towerKills'], 'o')

plt.show()

# 10) Data clustering

dfn = pd.read_csv("games.csv")
droper(dfn)
norm_dfn = (dfn - dfn.mean()) / dfn.std()
norm_dfn.drop('gameId', inplace=True, axis=1)
norm_dfn.drop('creationTime', inplace=True, axis=1)
norm_dfn.drop('gameDuration', inplace=True, axis=1)
norm_dfn.drop('seasonId', inplace=True, axis=1)
print(norm_dfn)
