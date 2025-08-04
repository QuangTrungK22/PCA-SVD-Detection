import numpy as np

class MySVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.Vt = None

    def fit(self,X):
        self.mean = np.mean(X , axis = 0)
        X_centered = X - self.mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.Vt = Vt[:self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.Vt.T
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        