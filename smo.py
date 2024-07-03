from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt



class SVM:
    def __init__(self, kernel='linear', C=1, max_iter=10000, tau=1e-3, eps=1e-2, degree=2, coef=0.0, gamma=1.0):
        """
        Sequentiel Minimal Optimization (SMO) algorithm for training SVMs.

        Args:
            kernel (str): kernel function to use. Defaults to 'linear'.
            C (float): regularization parameter
            max_iter (int): number of passes. Defaults to 1000.
            tau (float): positive tolerance parameters.
        """
        self.kernel = kernel 
        self.C = C
        self.max_iter = max_iter
        self.tau = tau
        self.eps = eps
        self.degree = degree # for polynomial kernel
        self.coef = coef # for polynomial/sigmoid kernel
        self.gamma = gamma # for RBF kernel
        

    def select_violated_pair(self, n) :
        """
        Function to select the maximum pair of the Lagrange multipliers that violated the KKT conditions

        Args:
            n (int) : number of samples 
        Return:
            Index for the pair of the Lagrange multipliers
        """
        I_all = np.arange(n)

        pass


    def compute_bounds(self, y1, y2, alpha1, alpha2, C):
        """
        Function to compute the bounds L and H 

        Args:
            y1 (int) :  labels for alpha1
            y2 (int) :  labels for alpha2
            alpha1 () : Lagrange multipliers 1
            alpha2 () : Lagrange multipliers 2
            C (int) : bounds

        Return:
            The bounds L and H
        """
        if (y1 == y2) :
            L = np.max(0, alpha1 + alpha2 - C)
            H = np.min(C, alpha1 + alpha2)
        else :
            L = np.max(0, alpha2 - alpha1)
            H = np.min(C, C + alpha2 + alpha1)

        return L, H

    def update_threshold(self, b):
        """
        Function to update the threshold b after the update of the aplhas

        Args:
            b () : threshold
        
        Return:
            Update version of b that respect the KKT conditions
        """
        pass

    def compute_f(self):
        """
        Function to compute the objective function

        Args:

        Return:
            Objective function
        """
        pass

    def compute_F(self):
        """
        """
        pass

    def compute_eta(self, X, i, j):
        """
        Function to compute eta

        Args:
            X : Training data features
            i : index of the Lagrange multiplier 1
            j : index of the Lagrange multiplier 2

        Return:
            Value of the second derivative of the objective function
        """
        if self.kernel == 'linear':
            K_ij = np.dot(X[i], X[j])
            K_ii = np.dot(X[i], X[i])
            K_jj = np.dot(X[j], X[j])
        elif self.kernel == 'poly':
            K_ij = (np.dot(X[i], X[j]) + self.coef0) ** self.degree
            K_ii = (np.dot(X[i], X[i]) + self.coef0) ** self.degree
            K_jj = (np.dot(X[j], X[j]) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            K_ij = np.exp(-self.gamma * np.linalg.norm(X[i] - X[j]) ** 2)
            K_ii = np.exp(-self.gamma * np.linalg.norm(X[i] - X[i]) ** 2)
            K_jj = np.exp(-self.gamma * np.linalg.norm(X[j] - X[j]) ** 2)
            K_ii = K_jj = 1.0  # Simplified because distance is zero
        else :
            raise ValueError("Unsupported kernel type: {}".format(self.kernel))
        
        eta = 2 * K_ij - K_ii - K_jj
        return eta

    def fit(self, X_train, y_train):
        """
        Function to train the algorithm
        """
        n, m = X_train.shape
        # Initialisation of the alphas
        self.alphas = np.zeros(n)

        for i in range(self.max_iter):
            # Compute of the index for both alphas
            index_alpha_1, index_alpha_2 = self.select_violated_pair(n)
            
            # Compute of the bounds
            L, H = self.compute_bounds(self.y_train[index_alpha_1], self.y_train[index_alpha_2], self.alphas[index_alpha_1], self.alphas[index_alpha_2], self.C)
            if L == H :
                continue # to next i
            
            # Compute of eta
            eta = self.compute_eta(X_train, index_alpha_1, index_alpha_2)
