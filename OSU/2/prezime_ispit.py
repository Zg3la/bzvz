#import bibilioteka
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from tensorflow import keras
from keras import layers

##################################################
#1. zadatak
##################################################

#učitavanje dataseta

"""data = pd.read_csv('titanic.csv')

#a)
print(f'Women: {data[data['Sex'] == 'female'].shape[0]}')

#b)
print(f'Died: {data[data['Survived'] == 0].shape[0] / data.shape[0]}%')

#c)
men_survived = data[(data['Survived'] == 1) & (data['Sex'] == 'male')]
women_survived = data[(data['Survived'] == 1) & (data['Sex'] == 'female')]

y = [men_survived.shape[0] / data[data['Sex'] == 'male'].shape[0],
     women_survived.shape[0] / data[data['Sex'] == 'female'].shape[0]]
x = ['male', 'female']
plt.bar(x, y, color = ['green', 'yellow'])
plt.xlabel('spol')
plt.ylabel('postotak')
plt.title("Postotak prezivjelih prema spolu")
#plt.show()

# postotak zena koje su prezivjele je puno veci

#d)

print(f"Average survived male age: {men_survived['Age'].mean()}")
print(f"Average survived female age: {women_survived['Age'].mean()}")

#e)

classes = men_survived['Pclass'].unique()

print(classes)
classes.sort()

for x in classes:
    print(f'Class {x}: {men_survived[men_survived["Pclass"] == x]['Age'].max()}')

# u nizim klasama su umrli stariji ljudi
"""
##################################################
#2. zadatak
##################################################

#učitavanje dataseta
"""data = pd.read_csv('titanic.csv')

data.dropna(inplace=True)

data['Sex'] = data['Sex'].astype('category').cat.codes
data['Pclass'] = data['Pclass'].astype('category').cat.codes
data['Embarked'] = data['Embarked'].astype('category').cat.codes


x = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
y = data['Survived']


#train test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify=y, random_state = 10)

# skaliranje
sc = StandardScaler()
x_train_n = x_train

fare_column = x_train_n['Fare'].values.reshape(-1, 1)

scaled_fare = sc.fit_transform(fare_column)
x_train_n['Fare'] = scaled_fare.flatten()

# isto ko i za x_train_n sam skraceno
x_test_n = x_test
x_test_n['Fare'] = sc.transform(x_test['Fare'].values.reshape(-1,1)).flatten()

#a)
def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)

KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(x_train_n, y_train)
y_test_p = KNN_model.predict(x_test)
y_train_p = KNN_model.predict(x_train_n)


#plot_decision_regions(x_train_n[['Pclass', 'Sex']], y_train, classifier=KNN_model)

#b)

print("KNN klasifikacija (k=5): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))


#c)

param_grid = {'n_neighbors': np.arange(1,10)}

gscv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5, scoring ='accuracy', n_jobs =-1)
gscv.fit(x_train_n, y_train)
print('KNN grid search result: ' + str(gscv.best_params_))
#print(gscv.best_score_)
#print(gscv.cv_results_)

#d)

KNN_model = KNeighborsClassifier(n_neighbors = 4)
KNN_model.fit(x_train_n, y_train)
y_test_p = KNN_model.predict(x_test)
y_train_p = KNN_model.predict(x_train_n)

print("KNN klasifikacija (k=4): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))
"""

##################################################
#3. zadatak
##################################################

#učitavanje podataka: (ponovljeno iz drugog zadatka)
data = pd.read_csv('titanic.csv')

data.dropna(inplace=True)

data['Sex'] = data['Sex'].astype('category').cat.codes
data['Pclass'] = data['Pclass'].astype('category').cat.codes
data['Embarked'] = data['Embarked'].astype('category').cat.codes


x = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
y = data['Survived']


#train test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, stratify=y, random_state = 1)

# skaliranje
sc = StandardScaler()
x_train_n = x_train

fare_column = x_train_n['Fare'].values.reshape(-1, 1)

scaled_fare = sc.fit_transform(fare_column)
x_train_n['Fare'] = scaled_fare.flatten()

# isto ko i za x_train_n sam skraceno
x_test_n = x_test
x_test_n['Fare'] = sc.transform(x_test['Fare'].values.reshape(-1,1)).flatten()


#a)

model = keras.Sequential()
model.add(layers.Input(shape = (4, )))
model.add(layers.Dense(12, activation ="relu"))
model.add(layers.Dense(8, activation ="relu"))
model.add(layers.Dense(4, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
print(model.summary())

#b)

model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy",])

#c)

model.fit(x_train, y_train, batch_size=5, epochs=100, validation_split=0.1)

#d)

model.save("model.keras")

del model

#e)

model = keras.models.load_model("model.keras")

score = model.evaluate( x_test, y_test, verbose =0)
print("Accuracy: ", score[1])
print("Loss: ", score[0])

#f)
predicted = model.predict(x_test)
predicted[predicted > 0.5] = 1
predicted[predicted <= 0.5] = 0
print("Confusion matrix: \n", confusion_matrix(y_test, predicted))
