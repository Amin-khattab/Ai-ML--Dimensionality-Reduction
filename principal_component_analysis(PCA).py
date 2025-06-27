import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from  sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

DR = PCA(n_components=2)
x_train_scaled = DR.fit_transform(x_train_scaled)
x_test_scaled = DR.transform(x_test_scaled)

classifier =LogisticRegression(random_state=0)
classifier.fit(x_train_scaled, y_train)
y_pred = classifier.predict(x_test_scaled)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


X_set, y_set = x_train_scaled, y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)
plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue')))
for i, color in zip(np.unique(y_set), ('red', 'green', 'blue')):
    plt.scatter(X_set[y_set == i, 0], X_set[y_set == i, 1],
                c=color, label=f'Class {i}')
plt.title('Logistic Regression (Training set, PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


X_set, y_set = x_test_scaled, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)
plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue')))
for i, color in zip(np.unique(y_set), ('red', 'green', 'blue')):
    plt.scatter(X_set[y_set == i, 0], X_set[y_set == i, 1],
                c=color, label=f'Class {i}')
plt.title('Logistic Regression (Test set, PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
