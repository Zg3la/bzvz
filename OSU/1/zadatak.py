import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from tensorflow import keras
from keras import layers

# ZADATAK 1
"""
data = pd.read_csv('diabetes.csv')
# a)
print(f"Obavljeno je {data.shape[0]} mjerenja")

# b)
print(f"Postoji li izostale ili duplicirane: {data[['BMI', 'Age']].isnull().values.any() or data[['BMI', 'Age']].duplicated().any()}")

data.drop_duplicates(['BMI', 'Age'], inplace=True)
data.dropna(subset=['BMI', 'Age'], inplace=True)

print(f"Ostalo je {data.shape[0]} uzoraka")


# c)

plt.scatter(data['BMI'], data['Age'])
plt.title('BMI vs Age')
plt.xlabel('BMI')
plt.ylabel('Age')
plt.show()

# BMI ostaje slican za sve dobne skupine, ali u starijim dobima nema ljudi sa visokim BMI-jem


# d)
print(f"BMI\nMin: {data['BMI'].min()}\nMax: {data['BMI'].max()}\nMean: {data['BMI'].mean()}")


# e)

print(f"\nBMI (sa dijabetesom)\nMin: {data[data['Outcome'] == 1]['BMI'].min()}\n"
      + f"Max: {data[data['Outcome'] == 1]['BMI'].max()}\n"
      + f"Mean: {data[data['Outcome'] == 1]['BMI'].mean()}")

print(f"\nBMI (bez dijabetesa)\nMin: {data[data['Outcome'] == 0]['BMI'].min()}\n"
      + f"Max: {data[data['Outcome'] == 0]['BMI'].max()}\n"
      + f"Mean: {data[data['Outcome'] == 0]['BMI'].mean()}")

print(f"Broj ljudi sa dijabetesom: {data[data['Outcome'] == 1].shape[0]}\n")
"""

# ZADATAK 2

"""
data = pd.read_csv('diabetes.csv')

X = data.drop('Outcome', axis=1)
y = data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a)
LogRegression_model = LogisticRegression()
LogRegression_model.fit(x_train , y_train)

# b)
y_test_p = LogRegression_model.predict(x_test)

# c)
cm = confusion_matrix(y_test, y_test_p)
print(cm)

# d)
# tocnost
print (" Tocnost : ", accuracy_score(y_test, y_test_p))
# report
print(classification_report(y_test, y_test_p))
"""

# ZADATAK 3

data = pd.read_csv('diabetes.csv')

X = data.drop('Outcome', axis=1)
y = data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a)

model = keras.Sequential()
model.add(layers.Input(shape = (8, )))
model.add(layers.Dense(12, activation ="relu"))
model.add(layers.Dense(8, activation ="relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
print(model.summary())

# b)

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy",])

# c)

model.fit(x_train, y_train, batch_size=10, epochs=150, validation_split=0.1)

# d)

model.save("model.keras")

del model

model = keras.models.load_model("model.keras")

# e)

score = model.evaluate( x_test, y_test, verbose =0)
print("Accuracy: ", score[1])
print("Loss: ", score[0])

# f)

predicted = model.predict(x_test)
predicted = [1 if x > 0.5 else 0 for x in predicted]

print("Confusion matrix: \n", confusion_matrix(y_test, np.argmax(model.predict(x_test), axis=1)))
