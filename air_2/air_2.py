import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

symptoms_df = pd.read_csv('Symptom-severity.csv')
symptoms_df.head()

len(symptoms_df)

df = pd.read_csv('dataset.csv')
df

X = df.drop('Disease', axis=1).to_numpy()
y = df['Disease'].to_numpy()

symptoms = symptoms_df['Symptom'].to_numpy()
symptom_dic = {}
for i in range(len(symptoms)):
  symptom_dic[symptoms[i]] = i

X_new = np.zeros((len(X),len(symptoms)))

for i in range(len(X)):
  for j in range(len(X[i])):
    if X[i][j] is np.nan: break
    key = ''.join(X[i][j].split(' '))
    X_new[i][symptom_dic[key]] = 1

disease_dict = {}
for i, d in enumerate(df['Disease'].unique()):
  disease_dict[d] = i

y_new = np.array([disease_dict[d] for d in y])
y_new = tf.one_hot(y_new, np.max(y_new+1)).numpy()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(41, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.1)

X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

model.fit(X_train, y_train, batch_size=32, epochs=5)

preds = model.predict(X_test)

y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_test, axis=1)

print(f'Accuracy: {accuracy_score(y_true, y_pred)}')