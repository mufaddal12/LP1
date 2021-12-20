import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns

iris_df = pd.read_csv("Iris.csv").drop('Id', axis=1)

iris_df.head()

iris_df.shape

iris_df.columns

iris_df.describe(include='all')

iris_df.describe()

sns.histplot(data=iris_df, x = 'SepalLengthCm')

sns.histplot(data=iris_df, x = 'SepalWidthCm')

sns.histplot(data=iris_df, x = 'PetalLengthCm')

sns.histplot(data=iris_df, x = 'PetalWidthCm')

sns.boxplot(data=iris_df)

