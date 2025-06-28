import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA

dataset = pd.read_csv("Wine.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

DR = KernelPCA(n_components=2, kernel='rbf', random_state=0)
x_train_scaled = DR.fit_transform(x_train_scaled)
x_test_scaled = DR.transform(x_test_scaled)

classiefier = LogisticRegression(random_state=0)
classiefier.fit(x_train_scaled,y_train)
y_pred = classiefier.predict(x_test_scaled)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


def plot_decision_boundary(X, y, classifier, title):
    from matplotlib.colors import ListedColormap
    X1, X2 = X[:, 0], X[:, 1]
    x1_min, x1_max = X1.min() - 1, X1.max() + 1
    x2_min, x2_max = X2.min() - 1, X2.max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                         np.arange(x2_min, x2_max, 0.01))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.scatter(X1, X2, c=y, s=40, edgecolor='k', cmap=ListedColormap(('red', 'green', 'blue')))
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


plot_decision_boundary(x_train_scaled, y_train, classiefier, 'Kernel PCA (Train set)')

plot_decision_boundary(x_test_scaled, y_test, classiefier, 'Kernel PCA (Test set)')
