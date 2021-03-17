import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import seaborn as sns
from typing import Tuple, List
import re
import datetime


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))



df = pd.read_csv("games.csv")
print(df.shape)
print(df.columns)
columna=df['gameDuration']
print(columna)
print_tabulate(df)

df_aux=df.groupby(["gameId","creationTime"])[['gameDuration']]
df_aux.plot(y='gameDuration',legend=False,figsize=(32,18))
plt.xticks(rotation=90)
plt.savefig("foo.png")
plt.close()
