from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt



class SVM:
    def __init__(self, kernel='linear', C=1, max_iter=10000, tau=1e-3, eps=1e-2, degree=2, coef0=0.0, gamma=1.0):
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
        self.coef0 = coef0 # for polynomial/sigmoid kernel
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

    def compute_objective_function(self, X, y, alphas):
        """
        Function to compute the objctive function

        Args:
            X : Training data features
            y : labels
            alphas (array) : Lagrange multipliers
        """
        n_samples = X.shape[0]
        # Term 1: 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
        term1 = 0.0
        for i in range(n_samples):
            for j in range(n_samples):
                term1 += self.alphas[i] * self.alphas[j] * y[i] * y[j] * self.kernel(X[i], X[j])
        term1 *= 0.5
        
        # Term 2: sum(alpha_i)
        term2 = np.sum(self.alphas)
        
        # Objective function value
        objective_value = term1 - term2
        return objective_value

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
            K_ii = K_jj = 1.0  
        else :
            raise ValueError("Unsupported kernel type: {}".format(self.kernel))
        
        eta = 2 * K_ij - K_ii - K_jj
        return eta
    
    def update_alphas(self, ind1, ind2, y1, y2, E_1, E_2, eta, L, H):
        """
        Function to update the alphas

        Args:
            ind1 : index of the Lagrange multipliers 1
            ind2 : index of the Lagrange multipliers 2
            y1 : label of the Lagrange multipliers 1
            y2 : label of the Lagrange multipliers 2
            E_1 : Error 1
            E_2 : Error 2
            eta : second derivative of the objective function
            L, H : bounds

        """
        # store old alpha2
        old_alpha2 = self.alphas[ind2]
        # alpha_2_new
        self.alphas[ind2] =  self.alphas[ind2] - (y2 * (E_1 - E_2) / eta) 
        
        # alpha_2_new_clipped
        if self.alphas[ind2] >= H :
            self.alphas[ind2] = H
        elif self.alphas[ind2] <= L :
            self.alphas[ind2] = L

        # alpha_1_new 
        self.alphas[ind1] = self.alphas[ind1] + y1*y2 * (old_alpha2 - self.alphas[ind2])

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
            L, H = self.compute_bounds(y_train[index_alpha_1], y_train[index_alpha_2], self.alphas[index_alpha_1], self.alphas[index_alpha_2], self.C)
            if L == H :
                continue # to next i
            
            # Compute of eta
            eta = self.compute_eta(X_train, index_alpha_1, index_alpha_2)
            if eta >= 0 :
                continue  # to next i
            
            # Compute of the objective function f
            f = self.compute_objective_function(self, X_train, y_train, self.alphas)

            # Compute of the Error E_i which is the error between the SVM output on the ith example and the true label
            E_1 = f[X_train[index_alpha_1]] - y_train[index_alpha_1]
            E_2 = f[X_train[index_alpha_2]] - y_train[index_alpha_2]

            # Update of the lagrange multipliers
            self.update_alphas(self, index_alpha_1, index_alpha_2, y_train[index_alpha_1], y_train[index_alpha_2], E_1, E_2, eta, L, H)
