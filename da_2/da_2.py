import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics

df = pd.read_csv('diabetes.csv')
df.head()

train_df, test_df = train_test_split(df, test_size = 0.1)

train_X, train_y = train_df.drop('Outcome', axis=1), train_df['Outcome']
test_X, test_y = test_df.drop('Outcome', axis=1), test_df['Outcome']

class NaiveBayesClassifier():
  def __separate(self, X, y):
    separated = [[] for _ in range(self.num_classes)]
    for i in range(len(X)):
      separated[y[i]].append(X[i])
    return separated

  def __calc_prob(self, x, mean, std):
    expo = np.exp(-((x-mean)**2)/(2 * std**2))
    return (1 / (np.sqrt(2*np.pi) * std)) * expo

  def __calculate_class(self, x):
    probs = [1] * self.num_classes
    for i in range(len(self.summaries)):
      for j in range(len(x)):
        probs[i] *= self.__calc_prob(x[j], self.summaries[i]['mean'][j],
                                     self.summaries[i]['std'][j])
    return np.argmax(probs)

  def fit(self, X, y):
    self.num_classes = np.max(y) + 1
    separated = self.__separate(X, y)
    self.summaries = []
    for i in range(len(separated)):
      self.summaries.append({
          'mean': np.mean(separated[i], axis = 0),
          'std': np.std(separated[i], axis = 0)
      })

  def predict(self, X):
    preds = []
    for x in X:
      preds.append(self.__calculate_class(x))
    return preds

clf = NaiveBayesClassifier()

clf.fit(train_X.to_numpy(), train_y.to_numpy())

preds = clf.predict(test_X.to_numpy())

print(metrics.classification_report(test_y, preds))

