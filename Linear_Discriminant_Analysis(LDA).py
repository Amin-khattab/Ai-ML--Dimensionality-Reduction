import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,confusion_matrix
from matplotlib.colors import ListedColormap

dataset = pd.read_csv("Wine.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

lda = LinearDiscriminantAnalysis(n_components=2)
x_train_scaled = lda.fit_transform(x_train_scaled, y_train)
x_test_scaled = lda.transform(x_test_scaled)

classifier = LogisticRegression()
classifier.fit(x_train_scaled,y_train)
y_pred = classifier.predict(x_test_scaled)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


def plot_decision_boundary(X, y, classifier, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
        np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(
        X1, X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).reshape(2, -1).T).reshape(X1.shape),
        alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue'))
    )
    for i, color in zip(np.unique(y), ('red', 'green', 'blue')):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=color, label=f'Class {i}')
    plt.title(title)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend()
    plt.show()


plot_decision_boundary(x_train_scaled, y_train, classifier, 'Logistic Regression (Training set)')


plot_decision_boundary(x_test_scaled, y_test, classifier, 'Logistic Regression (Test set)')
