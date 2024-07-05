import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import smo

iris = load_iris()
X = iris.data[50:, 2:]
Y = iris.target[50:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
svm = smo.SVM(kernel='linear', C=1)
svm.fit(X_train, Y_train)

print(f'self.b: {svm.b}')

Y_pred = svm.predict(X_test)
accuracy = np.mean(Y_pred == Y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
