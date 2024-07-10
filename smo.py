import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

class SVM:
    def __init__(self, kernel='linear', C=float('-inf'), max_iter=3000, tau=1e-3, eps=1e-10, degree=2, coef0=0.0, gamma=1.0, tol=1e-5):
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
        self.tol = tol
        self.objective_values = []
        
    
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

    def select_violated_pair(self, n, K):
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
        

        F = self.compute_F(self.X_train, K)
        

        max_diff = -np.inf
        index_alpha_1, index_alpha_2 = -1, -1
         
        for i in I_up:
            for j in I_low:
                if F[i] < F[j] - self.tau:
                    violation = F[j] - F[i] - self.tau
                    if violation > max_diff:
                        max_diff = violation
                        index_alpha_1, index_alpha_2 = i, j
        
        if max_diff == -np.inf:
            return -1, -1
        
        
        return index_alpha_1, index_alpha_2
        
        # # if max_diff < self.tau:
        # #     return -1, -1
        # print(f'index_alpha_1: {index_alpha_1}, index_alpha_2: {index_alpha_2}')
        # return I_up[index_alpha_1], I_low[index_alpha_2]
        
        # violating_matrix = F[I_up][:, None] < F[I_low] - self.tau # matrice booleenne 
        # violating_values = np.where(violating_matrix, F[I_low] - F[I_up][:, None] - self.tau, -float("inf")) # matrice de valeurs
        # # Find the maximum violating pair
        # max_index = np.unravel_index(np.argmax(violating_values), violating_values.shape)
        # max_violating_value = violating_values[max_index]

        # if max_violating_value == -float("inf"): # si aucune paire ne satisfait la condition de violation
        #     return -1, -1
        
        # return I_up[max_index[0]], I_low[max_index[1]] # indices de la paire maximale violante


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
    
    def compute_F(self, X, K):
        
        Y = np.outer(self.y_train, self.y_train)   
        F = self.y_train * (np.sum(np.dot(Y * K, np.diag(self.alphas)), axis=1) - 1)
       
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
        # print("APLHAS : ", self.alphas)
        # K = self.compute_kernel(self.X_train, self.X_train)
        # term1 = 0.5 * np.sum((self.alphas * self.y_train)[:, None] * (self.alphas * self.y_train)[None, :] * K)
        # term2 = np.sum(self.alphas)
        # return term1 - term2
        
    def update_alphas(self, index_alpha_1, index_alpha_2, E_1, E_2, eta, L, H, K):
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
            print("eta >= 0, on quitte")
            return False

        # alpha_j_new = [alpha_j_old - self.y_train[index_alpha_2] * (E_1 - E_2) / eta][0][0]
        
        # if alpha_j_new >= H:
        #     alpha_j_new = H
        # elif alpha_j_new <= L:
        #     alpha_j_new = L 
        
        

        w = self.alphas * self.y_train 
       
        if self.y_train[index_alpha_1] == self.y_train[index_alpha_2]:
            v2 = (self.y_train[index_alpha_1] * (np.dot(w, K[index_alpha_1]) - np.dot(w, K[index_alpha_2]))) / eta
        else :
            v2 = (2 + self.y_train[index_alpha_1] * (np.dot(w, K[index_alpha_1]) - np.dot(w, K[index_alpha_2]))) / eta
      
        alpha_j_new2 = -v2 + self.alphas[index_alpha_2]
        alpha_j_new2 = min(H, max(L, alpha_j_new2))

        print("alpha_q_new_", alpha_j_new2)
        
        
        alpha_i_new = alpha_i_old + self.y_train[index_alpha_1] * self.y_train[index_alpha_2] * (alpha_j_old - alpha_j_new2)
        
        # print("p", index_alpha_1, "q", index_alpha_2)
        # print("alpha_p_new", alpha_i_new, "alpha_q_new", alpha_j_new)
        
        print("alpha_p_new", alpha_i_new)
        print("---------------------")
        self.alphas[index_alpha_1] = alpha_i_new
        self.alphas[index_alpha_2] = alpha_j_new2
         
        return True
        


    # def update_threshold(self, index_alpha_1, index_alpha_2, E_1, E_2, alpha_i_old, alpha_j_old):
    #     """
    #     Update the threshold value b.

    #     Parameters:
    #     index_alpha_1: int
    #         The index of the first sample.
    #     index_alpha_2: int
    #         The index of the second sample.
    #     E_1: float
    #         The error of the first sample.
    #     E_2: float
    #         The error of the second sample.
    #     alpha_i_old: float
    #         The old value of the first alpha.
    #     alpha_j_old: float
    #         The old value of the second alpha.
    #     """
    #     b1 = self.b - E_1 - self.y_train[index_alpha_1] * (self.alphas[index_alpha_1] - alpha_i_old) * self.compute_kernel(self.X_train[index_alpha_1], self.X_train[index_alpha_1]) - \
    #          self.y_train[index_alpha_2] * (self.alphas[index_alpha_2] - alpha_j_old) * self.compute_kernel(self.X_train[index_alpha_1], self.X_train[index_alpha_2])
        
    #     b2 = self.b - E_2 - self.y_train[index_alpha_1] * (self.alphas[index_alpha_1] - alpha_i_old) * self.compute_kernel(self.X_train[index_alpha_1], self.X_train[index_alpha_2]) - \
    #          self.y_train[index_alpha_2] * (self.alphas[index_alpha_2] - alpha_j_old) * self.compute_kernel(self.X_train[index_alpha_2], self.X_train[index_alpha_2])
        
    #     if 0 < self.alphas[index_alpha_1] < self.C:
    #         self.b = b1
    #     elif 0 < self.alphas[index_alpha_2] < self.C:
    #         self.b = b2
    #     else:
    #         self.b = (b1 + b2) / 2
        
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
        self.min_val = np.min(labels)
        self.max_val = np.max(labels)
        return np.where(labels == self.min_val, -1, 1)
    
    
    def get_support_vectors(self):
        ind_sv = np.where((self.eps < self.alphas) & (self.alphas < self.C))[0] # tableau des indices, retourne le 1er element du tableau
        ind_inner = np.where(self.alphas == self.C)[0]

        return ind_sv, ind_inner
    
    
    def plot_data(self):
        x_min, x_max = self.X_train[:, 0].min() - 0.5, self.X_train[:, 0].max() + 0.5
        y_min, y_max = self.X_train[:, 1].min() - 0.5, self.X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        plt.scatter(self.X_train[self.y_train == min(self.y_train), 0], self.X_train[self.y_train == min(self.y_train), 1], c='red', edgecolor='k', s=20, label='Setosa')
        plt.scatter(self.X_train[self.y_train == max(self.y_train), 0], self.X_train[self.y_train == max(self.y_train), 1], c='blue', edgecolor='k', s=20, label='Versicolor')
        plt.scatter(self.X_train[self.ind_sv, 0], self.X_train[self.ind_sv, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Petal Width (cm)')
        plt.legend()
        plt.show()

    def plt_Data_and_Boundary(self):
        """
        Function to plot data points and decision boundaries
        """
        x_min, x_max = self.X_train[:, 0].min() - 0.5, self.X_train[:, 0].max() + 0.5
        y_min, y_max = self.X_train[:, 1].min() - 0.5, self.X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        plt.scatter(self.X_train[self.y_train == min(self.y_train), 0], self.X_train[self.y_train == min(self.y_train), 1], c='red', edgecolor='k', s=20, label='Setosa')
        plt.scatter(self.X_train[self.y_train == max(self.y_train), 0], self.X_train[self.y_train == max(self.y_train), 1], c='blue', edgecolor='k', s=20, label='Versicolor')
        plt.scatter(self.X_train[self.ind_sv, 0], self.X_train[self.ind_sv, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

        XY = np.vstack([xx.ravel(), yy.ravel()]).T
        Z = self.predict(XY).reshape(xx.shape)
        #Z = np.array([self.predict(np.array([xx.ravel()[i], yy.ravel()[i]]).reshape(1, -1)) for i in range(len(xx.ravel()))])
        levels = [self.min_val, (self.min_val + self.max_val) / 2, self.max_val]
        plt.contourf(xx, yy, Z, levels, alpha=0.3, colors=['pink', 'lightblue'])
        plt.contour(xx, yy, Z, levels= [(self.min_val + self.max_val) / 2], colors='k', linestyles='-')

        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Petal Width (cm)')
        plt.legend()
        plt.show()

        
        
    def plt_Objective_Values(self):
        
        plt.plot(self.objective_values, label='objective function\'s value')
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.legend()
        plt.show()





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
        
        K = self.compute_kernel(X_train, self.X_train)
        
        for _ in (range(self.max_iter)):
            
            index_alpha_1, index_alpha_2 = self.select_violated_pair(n, K)
               
            # print("index_1", index_alpha_1, "index_2", index_alpha_2)
        
            previous_objective_value =  self.objective_values[-1] if len(self.objective_values) > 0 else 0.0
            objective_value = self.compute_objective_function()
            self.objective_values.append(objective_value)

            # print(f'Obejctive function value: {objective_value}')
      
            
            if (index_alpha_1 == -1 or index_alpha_2 == -1) or (_ > 1 and previous_objective_value == objective_value):
                # print("index_1", index_alpha_1, "index_2", index_alpha_2)
                # print("previous_objective_value", previous_objective_value, "objective_value", objective_value)
                print("Converged")
                break
            
            
            eta = self.compute_eta(self.X_train, index_alpha_1, index_alpha_2)
            
            L, H = self.compute_bounds(index_alpha_1, index_alpha_2)
            if L == H:
                print("we continue here1")
                continue
           
            E_1 = self.compute_F(self.X_train[index_alpha_1].reshape(1, -1), K) - self.y_train[index_alpha_1]
            E_2 = self.compute_F(self.X_train[index_alpha_2].reshape(1, -1), K) - self.y_train[index_alpha_2]

            alpha_1_old, alpha_2_old = self.alphas[index_alpha_1], self.alphas[index_alpha_2]
            
            if not self.update_alphas(index_alpha_1, index_alpha_2, E_1, E_2, eta, L, H, K):
                print("we continue here2")
                continue
            
            # print(f'alpha_1_new: {self.alphas[index_alpha_1]}, alpha_2_new: {self.alphas[index_alpha_2]}')
            
            # self.update_threshold(index_alpha_1, index_alpha_2, E_1, E_2, alpha_1_old, alpha_2_old)

            
         
        self.b = (1 / self.y_train[index_alpha_1]) - np.sum(self.alphas * self.y_train * K[:, index_alpha_1])
        
        self.ind_sv, self.ind_inner = self.get_support_vectors()
        
        # self.b = np.sum(self.y_train[self.ind_sv])
        # for i in self.ind_sv:
        #     for j in range(n):
        #         self.b -= self.alphas[j] * self.y_train[j] * self.compute_kernel(self.X_train[j], self.X_train[i])
        # self.b /= len(self.ind_sv)

        # print(f'Number of iterations: {_}')
        # print(f'Objective function value: {objective_value}')
        # print(f'Support vectors: {np.sum(self.alphas > 1e-5)}')
        # print(f'Self.b: {self.b}')

        # print(f'ind_sv : {self.get_support_vectors()}')
        
    
    
    
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

        K = self.compute_kernel(X, self.X_train)
        return np.sign(np.sum(self.alphas * self.y_train * K, axis=1) + self.b)
    
    
    
    

# # Load dataset and prepare the data
# iris = load_iris()
# X = iris.data[50:, 2:]
# Y = iris.target[50:]

# # # Normalize the features
# # scaler = StandardScaler()
# # X = scaler.fit_transform(X)

# # Split the data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# # Train the SVM
# svm = SVM(kernel='linear', C=1)
# svm.fit(X_train, Y_train)


# # Evaluate the model
# Y_pred = svm.predict(X_test)
# accuracy = np.mean(Y_pred == svm.scale_labels(Y_test))
# print(f'Accuracy: {accuracy * 100:.2f}%')
# print(f'self.b: {svm.b}')

# print("Y_pred:", Y_pred)
# print("Y_test:", svm.scale_labels(Y_test))




# Load dataset and prepare the data
wine = load_wine()
X = wine.data[50:, 2:]
Y = wine.target[50:]

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
print(f'self.b: {svm.b}')

print("Y_pred:", Y_pred)
print("Y_test:", svm.scale_labels(Y_test))



svm.plt_Data_and_Boundary()
svm.plt_Objective_Values()
