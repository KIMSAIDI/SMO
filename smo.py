import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SVM:
    def __init__(self, kernel='linear', C=1, max_iter=10000, tau=1e-3, eps=1e-2, degree=2, coef0=0.0, gamma=1.0):
        """
        kernel: str, default='linear'
            Specifies the kernel type to be used in the algorithm.
            It must be one of 'linear', 'poly', 'rbf', 'sigmoid'.
        C: float, default=1
            Penalty parameter C of the error term.
        max_iter: int, default=10000
            The maximum number of iterations to run the algorithm.
        tau: float, default=1e-3
            Tolerance for the stopping criterion.
        eps: float, default=1e-2
            Tolerance for the numerical issues.
        degree: int, default=2
            Degree of the polynomial kernel function ('poly').
        coef0: float, default=0.0   
            Independent term in kernel function ('poly' and 'sigmoid').
        gamma: float, default=1.0   
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        """
        self.kernel = kernel 
        self.C = C
        self.max_iter = max_iter
        self.tau = tau
        self.eps = eps
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        
    
    def compute_kernel(self, x1, x2):
        """
        Compute the kernel function between two samples.

        Parameters:
        x1: numpy array
            The first sample.
        x2: numpy array
            The second sample.  

        Returns:    
        float
            The result of the kernel function  
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x1, x2.T) + self.coef0)
        else:
            raise ValueError("Unsupported kernel type: {}".format(self.kernel))

    def compute_eta(self, X, i, j):
        """
        Compute the second derivative of the objective function along the diagonal.

        Parameters:
        X: numpy array
            The input samples.
        i: int
            The index of the first sample.
        j: int
            The index of the second sample.

        Returns:
        float
            The second derivative of the objective function along the diagonal
        """
        K_ij = self.compute_kernel(X[i], X[j])
        K_ii = self.compute_kernel(X[i], X[i])
        K_jj = self.compute_kernel(X[j], X[j])
        
        if self.kernel == 'rbf':
            K_ii = 1.0
            K_jj = 1.0
        
        eta = 2 * K_ij - K_ii - K_jj
        return eta

    def select_violated_pair(self, n):
        """
        Select the pair of indices (index_alpha_1, index_alpha_2) that violate the KKT conditions.

        Parameters:
        n: int
            The number of samples.

        Returns:
        int
            The index of the first sample.
        """
        I_all = np.arange(n)
        I_0 = [i for i in I_all if (0 < self.alphas[i] < self.C)]
        I_1 = [i for i in I_all if (self.y_train[i] == 1 and self.alphas[i] == 0)]
        I_2 = [i for i in I_all if (self.y_train[i] == -1 and self.alphas[i] == self.C)]
        I_3 = [i for i in I_all if (self.y_train[i] == 1 and self.alphas[i] == self.C)]
        I_4 = [i for i in I_all if (self.y_train[i] == -1 and self.alphas[i] == 0)]

        I_up = I_0 + I_1 + I_2
        I_low = I_0 + I_3 + I_4

        F = self.compute_F(self.X_train)

        max_diff = -np.inf
        index_alpha_1, index_alpha_2 = -1, -1

        for i in I_up:
            for j in I_low:
                if F[i] - F[j] > max_diff:
                    max_diff = F[i] - F[j]
                    index_alpha_1, index_alpha_2 = i, j

        return index_alpha_1, index_alpha_2

    def compute_bounds(self, index_alpha_1, index_alpha_2):
        """
        Compute the bounds L and H for the alpha_2.

        Parameters:
        index_alpha_1: int
            The index of the first sample.
        index_alpha_2: int
            The index of the second sample.

        Returns:
        float
            The lower bound L.
        float
            The upper bound H.
        """

        if self.y_train[index_alpha_1] == self.y_train[index_alpha_2]:
            L = max(0, self.alphas[index_alpha_1] + self.alphas[index_alpha_2] - self.C)
            H = min(self.C, self.alphas[index_alpha_1] + self.alphas[index_alpha_2])
        else:
            L = max(0, self.alphas[index_alpha_2] - self.alphas[index_alpha_1])
            H = min(self.C, self.C + self.alphas[index_alpha_2] - self.alphas[index_alpha_1])

        return L, H
    
    def compute_F(self, X):
        """
        Compute the decision function F(x) = w^T x + b.

        Parameters:
        X: numpy array
            The input samples.

        Returns:
        numpy array
            The decision function F(x) = w^T x + b.
        """
        K = self.compute_kernel(X, self.X_train)  # Kernel matrix between X and X_train
        F = np.dot(K, self.alphas * self.y_train) + self.b  
        return F

    def compute_objective_function(self):
        """
        Compute the objective function.

        Returns:
        float
            The value of the objective function.
        """
        n_samples = self.X_train.shape[0]
        term1 = 0.0
        for i in range(n_samples):
            for j in range(n_samples):
                term1 += self.alphas[i] * self.alphas[j] * self.y_train[i] * self.y_train[j] * self.compute_kernel(self.X_train[i], self.X_train[j])
        term1 *= 0.5
        
        term2 = np.sum(self.alphas)
        
        objective_value = term1 - term2
        return objective_value

    def update_alphas(self, index_alpha_1, index_alpha_2, E_1, E_2, eta, L, H):
        """
        Update the alpha values based on the SMO algorithm.

        Parameters:
        index_alpha_1: int
            Index of the first sample.
        index_alpha_2: int
            Index of the second sample.
        E_1: float
            Error of the first sample.
        E_2: float
            Error of the second sample.
        eta: float
            Second derivative of the objective function along the diagonal.
        L: float
            Lower bound.
        H: float
            Upper bound.
            
        Returns:
        bool
            True if the alpha values were updated, False otherwise.
        """
        alpha_i_old, alpha_j_old = self.alphas[index_alpha_1], self.alphas[index_alpha_2]
        
        if eta >= 0:
            print("eta >= 0 : on part")
            return False

        alpha_j_new = alpha_j_old - self.y_train[index_alpha_2] * (E_1 - E_2) / eta
        alpha_j_new = max(L, min(alpha_j_new, H))
        
        # if abs(alpha_j_new - alpha_j_old) < self.eps:
        #     return False
        
        alpha_i_new = alpha_i_old + self.y_train[index_alpha_1] * self.y_train[index_alpha_2] * (alpha_j_old - alpha_j_new)
        
        self.alphas[index_alpha_1] = alpha_i_new
        self.alphas[index_alpha_2] = alpha_j_new

        # print("alpha_i_new:", alpha_i_new)  
        return True
        


    def update_threshold(self, index_alpha_1, index_alpha_2, E_1, E_2, alpha_i_old, alpha_j_old):
        """
        Update the threshold value b.

        Parameters:
        index_alpha_1: int
            The index of the first sample.
        index_alpha_2: int
            The index of the second sample.
        E_1: float
            The error of the first sample.
        E_2: float
            The error of the second sample.
        alpha_i_old: float
            The old value of the first alpha.
        alpha_j_old: float
            The old value of the second alpha.
        """
        b1 = self.b - E_1 - self.y_train[index_alpha_1] * (self.alphas[index_alpha_1] - alpha_i_old) * self.compute_kernel(self.X_train[index_alpha_1], self.X_train[index_alpha_1]) - \
             self.y_train[index_alpha_2] * (self.alphas[index_alpha_2] - alpha_j_old) * self.compute_kernel(self.X_train[index_alpha_1], self.X_train[index_alpha_2])
        
        b2 = self.b - E_2 - self.y_train[index_alpha_1] * (self.alphas[index_alpha_1] - alpha_i_old) * self.compute_kernel(self.X_train[index_alpha_1], self.X_train[index_alpha_2]) - \
             self.y_train[index_alpha_2] * (self.alphas[index_alpha_2] - alpha_j_old) * self.compute_kernel(self.X_train[index_alpha_2], self.X_train[index_alpha_2])
        
        if 0 < self.alphas[index_alpha_1] < self.C:
            self.b = b1
        elif 0 < self.alphas[index_alpha_2] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
    
    def scale_labels(self, labels):
        """
        Scale the labels to -1 and 1.

        Parameters:
        labels: numpy array
            The input labels.

        Returns:
        numpy array
            The scaled labels.
        """
        min_val = np.min(labels)
        max_val = np.max(labels)
        return np.where(labels == min_val, -1, 1)

    def fit(self, X_train, y_train):
        """
        Fit the model according to the given training data.

        Parameters:
        X_train: numpy array
            The input samples.
        y_train: numpy array
            The target values.
        """
        self.X_train = X_train
        self.y_train = self.scale_labels(y_train)
        n, m = X_train.shape
        
        self.alphas = np.zeros(n)
        self.b = 0.0

        for _ in range(self.max_iter):
            index_alpha_1, index_alpha_2 = self.select_violated_pair(n)
            if index_alpha_1 == -1 and index_alpha_2 == -1:
                print("Converged")
                break

            if _ % 100 == 0:
                print("self.alphas[index_alpha_1]:", self.alphas[index_alpha_1])
                print("self.alphas[index_alpha_2]:", self.alphas[index_alpha_2])
            L, H = self.compute_bounds(index_alpha_1, index_alpha_2)
            if L == H:
                continue
            
            E_1 = self.compute_F(self.X_train[index_alpha_1].reshape(1, -1)) - self.y_train[index_alpha_1]
            E_2 = self.compute_F(self.X_train[index_alpha_2].reshape(1, -1)) - self.y_train[index_alpha_2]

            eta = self.compute_eta(self.X_train, index_alpha_1, index_alpha_2)
            
            if not self.update_alphas(index_alpha_1, index_alpha_2, E_1, E_2, eta, L, H):
                continue

            self.update_threshold(index_alpha_1, index_alpha_2, E_1, E_2, self.alphas[index_alpha_1], self.alphas[index_alpha_2])

    
    
    def predict(self, X):
        """
        Perform classification on samples in X.

        Parameters:
        X: numpy array
            The input samples.

        Returns:
        numpy array
            The predicted classes.
        """
        F = self.compute_F(X)
        return np.sign(F)

# Load dataset and prepare the data
iris = load_iris()
X = iris.data[50:, 2:]
Y = iris.target[50:]

# # Normalize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Train the SVM
svm = SVM(kernel='linear', C=1)
svm.fit(X_train, Y_train)

# Evaluate the model
Y_pred = svm.predict(X_test)
accuracy = np.mean(Y_pred == svm.scale_labels(Y_test))
print(f'Accuracy: {accuracy * 100:.2f}%')
