import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

nltk.download('stopwords')

df = pd.read_csv('/content/drive/MyDrive/train_da_4.csv')
df

tokenizer = RegexpTokenizer(r'\w+')
X = [tokenizer.tokenize(phrase) for phrase in df['tweet']]
y = df['label']

print(X[0])

### **REMOVAL OF STOP WORDS**

def remove_stop_words(X):
  stop_words = stopwords.words('english')
  X_no_stop = [[word for word in X[i] if word not in stop_words] for i in range(len(X))]
  return X_no_stop

X_no_stop = remove_stop_words(X)

print(X_no_stop[0])

### **STEM WORDS IN DATASET**

def stem_words(X):
  ps = PorterStemmer()
  X_stemmed = [[ps.stem(word) for word in X[i]] for i in range(len(X))]
  return X_stemmed

X_stemmed = stem_words(X_no_stop)

print(X_stemmed[0])

### **TEXT VECTORIZATION**

def vectorize_text(X):
  X_joined = [' '.join(X[i]) for i in range(len(X))]
  vec = TfidfVectorizer()
  X_vectorized = vec.fit_transform(X_joined)
  return X_vectorized

X_vectorized = vectorize_text(X_stemmed)

print(X_vectorized[0])

### **TEXT CLASSIFICATION**

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.1, random_state=42)

X_train.shape

clf = RandomForestClassifier(random_state=42)

clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print(classification_report(y_test, preds))