from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt



class SMO:
    def __init__(self, kernel='linear', C=float('inf'), max_iter=10000, tau=1e-3):
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
    
    
    
    def fit(self):
        
        for iter in range(self.max_iter):
            # fit
            pass