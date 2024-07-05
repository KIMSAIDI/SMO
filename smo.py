from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import time



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
        
    
    def compute_kernel(self, x1, x2):
        """
        Function to compute the kernel function
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
        Function to compute eta

        Args:
            X : Training data features
            i : index of the Lagrange multiplier 1
            j : index of the Lagrange multiplier 2

        Return:
            Value of the second derivative of the objective function
        """
        K_ij = self.compute_kernel(X[i], X[j])
        K_ii = self.compute_kernel(X[i], X[i])
        K_jj = self.compute_kernel(X[j], X[j])
        
        if self.kernel == 'rbf':
            K_ii = 1.0
            K_jj = 1.0
        
        eta = 2 * K_ij - K_ii - K_jj
        return eta

    def select_violated_pair(self, n) :
        """
        Function to select the maximum pair of the Lagrange multipliers that violated the KKT conditions

        Args:
            n (int) : number of samples 
        Return:
            Index for the pair of the Lagrange multipliers
        """
        I_all = np.arange(n)
        I_0 = [i for i in I_all if ((self.alphas[i] > 0) & (self.alphas[i] < self.C))]
        I_1 = [i for i in I_all if ((self.y_train[i] == 1) & (self.alphas[i] == 0))]
        I_2 = [i for i in I_all if ((self.y_train[i] == -1) & (self.alphas[i] == self.C))]
        I_3 = [i for i in I_all if ((self.y_train[i] == 1) & (self.alphas[i] == self.C))]
        I_4 = [i for i in I_all if ((self.y_train[i] == -1) & (self.alphas[i] == 0))]

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
        Function to compute the bounds L and H 

        Args:
            index_alpha_1 (int): Index for alpha1
            index_alpha_2 (int): Index for alpha2

        Return:
            The bounds L and H
        """
        y1 = self.y_train[index_alpha_1]
        y2 = self.y_train[index_alpha_2]
        alpha1 = self.alphas[index_alpha_1]
        alpha2 = self.alphas[index_alpha_2]

        if y1 == y2:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)

        return L, H
    
    def compute_F(self, X):
        """
        Compute the decision function F for given input X.

        Args:
            X : Input data features

        Returns:
            Decision function values
        """
        K = self.compute_kernel(X, self.X_train)  # Kernel matrix between X and X_train
        F = np.dot(K, self.alphas * self.y_train) + self.b  
        return F
        

    def compute_objective_function(self):
        """
        Function to compute the objective function

        Return:
            Objective function value
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


    
    def update_alphas(self, i, j, E_i, E_j, eta, L, H):
        """
        Function to update the alphas

        Args:
            index_alpha_1 : index of the Lagrange multipliers 1
            index_alpha_1 : index of the Lagrange multipliers 2
            E_1 : Error 1
            E_2 : Error 2
            eta : second derivative of the objective function
            L, H : bounds
        """
        # y1 = self.y_train[index_alpha_1]
        # y2 = self.y_train[index_alpha_2]
        
        if eta >= 0:
            return False

        alpha_i_old = self.alphas[i]
        alpha_j_old = self.alphas[j]

        self.alphas[j] -= (self.y_train[j] * (E_i - E_j)) / eta
        self.alphas[j] = np.clip(self.alphas[j], L, H)

        if abs(self.alphas[j] - alpha_j_old) < self.eps:
            return False

        self.alphas[i] += self.y_train[i] * self.y_train[j] * (alpha_j_old - self.alphas[j])

        return True

        # # alpha_2_new
        # self.alphas[index_alpha_2] -= (y2 * (E_1 - E_2) / eta) 
        
        # # alpha_2_new_clipped
        # if self.alphas[index_alpha_2] >= H :
        #     self.alphas[index_alpha_2] = H
        # elif self.alphas[index_alpha_2] <= L :
        #     self.alphas[index_alpha_2] = L

        # # alpha_1_new 
        # self.alphas[index_alpha_1] += y1*y2 * (self.old_alpha2 - self.alphas[index_alpha_2])

    
    def update_threshold(self, ind1, ind2, E_1, E_2, y1, y2):
        """
        Function to update the threshold
        """
        b1 = self.b - E_1 - y1 * (self.alphas[ind1] - self.old_alpha1) * self.compute_kernel(self.X_train[ind1], self.X_train[ind1]) - y2 * (self.alphas[ind2] - self.old_alpha2) * self.compute_kernel(self.X_train[ind1], self.X_train[ind2])
        b2 = self.b - E_2 - y1 * (self.alphas[ind1] - self.old_alpha1) * self.compute_kernel(self.X_train[ind1], self.X_train[ind2]) - y2 * (self.alphas[ind2] - self.old_alpha2) * self.compute_kernel(self.X_train[ind2], self.X_train[ind2])
        
        if 0 < self.alphas[ind1] < self.C:
            self.b = b1
        elif 0 < self.alphas[ind2] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

    
    def scale_labels(self, labels):
        """
        Function to scale labels to -1 and 1.
        """
        min_val = np.min(labels)
        max_val = np.max(labels)
        
        return np.where(labels == min_val, -1, 1)

   

    def fit(self, X_train, y_train):
        """
        Function to train the algorithm
        """
        self.X_train = X_train
        self.y_train = self.scale_labels(y_train)
        n, m = X_train.shape
        
        # Initialisation of the alphas
        self.alphas = np.zeros(n)
        # Initialisation of the threshold
        self.b = 0.0

        for _ in range(self.max_iter):
            # Compute of the index for both alphas
            index_alpha_1, index_alpha_2 = self.select_violated_pair(n)
            if (index_alpha_1 == -1 or index_alpha_2 == -1):
                # has converge
                break
            
            # Compute of the bounds
            L, H = self.compute_bounds(index_alpha_1, index_alpha_2)
            if L == H :
                continue # to next i
            
            # Compute of eta
            # eta = self.compute_eta(self.X_train, index_alpha_1, index_alpha_2)
            # if eta >= 0 :
                #continue  # to next i
            
            E_1 = self.compute_F(self.X_train[index_alpha_1].reshape(1, -1)) - self.y_train[index_alpha_1]
            E_2 = self.compute_F(self.X_train[index_alpha_2].reshape(1, -1)) - self.y_train[index_alpha_2]

            eta = self.compute_eta(self.X_train, index_alpha_1, index_alpha_2)
            if not self.update_alphas(index_alpha_1, index_alpha_2, E_1, E_2, eta, L, H):
                continue
            
            """
            # Compute of the objective function f
            f = self.compute_objective_function()

            # Compute of the Error E_i which is the error between the SVM output on the ith example and the true label
            E_1 = f[X_train[index_alpha_1]] - y_train[index_alpha_1]
            E_2 = f[X_train[index_alpha_2]] - y_train[index_alpha_2]
            """

            # E_1 = self.compute_F(self.X_train)[index_alpha_1] - self.y_train[index_alpha_1]
            # E_2 = self.compute_F(self.X_train)[index_alpha_2] - self.y_train[index_alpha_2]

            # Update of the lagrange multipliers
            # self.old_alpha1 = self.alphas[index_alpha_1]
            # self.old_alpha2 = self.alphas[index_alpha_2]
            # self.update_alphas(index_alpha_1, index_alpha_2, E_1, E_2, eta, L, H)

            # Update of the threshold b 
            self.update_threshold(index_alpha_1, index_alpha_2, E_1, E_2, self.y_train[index_alpha_1], self.y_train[index_alpha_2])

   
    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Args:
            X : Input data features

        Returns:
            Predicted class labels
        """
        F = self.compute_F(X)  # Compute the decision function for the input data
        return np.sign(F)  # Return the sign of the decision function values
