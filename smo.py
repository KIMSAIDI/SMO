from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt



class SVM:
    def __init__(self, kernel='linear', C=1, max_iter=10000, tau=1e-3, eps=1e-2):
        """
        Sequentiel Minimal Optimization (SMO) algorithm for training SVMs.

        Args:
            kernel (str): kernel function to use. Defaults to 'linear'.
            C (float): regularization parameter
            max_iter (int): number of passes. Defaults to 1000.
            tau (float): positive tolerance parameters.
        """
        self.kernel = kernel # linear
        self.C = C
        self.max_iter = max_iter
        self.tau = tau
        self.eps = eps
        

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
            
