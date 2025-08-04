
import numpy as np
# đọc và tiền xử lý ảnh
def load_images(folder, img_size=(80,80)):
    img,labels = [], []
    for label_name in os.listdir(folder):
        subdir = os.path.join(folder, label_name)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            fpath = os.path.join(subdir, fname)
            img = cv2.imread(fpath,cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img,img_size).flatten()
            img.append(img)
            labels.append(label_name)
        return np.array(img), np.array(labels)
    
    # class PCA
class MyPCA:
    def __init__(self, n_components):
        self.n.components = n_components
        self.mean = None
        self.components = None

    def fit(self,X):
        self.mean = np.mean(X, axis =0)
        X_centered = X - self.mean    
        # coveariance matrix
        cov = (X_centered.T @ X_centered) / (X_centered.shape[0])
        # Eigen decomposition
        eigvals, eigvecs = np.linaglg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        self.components = eigvecs[:, idx[:self.n_components]]
    
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        # "N"
import cv2

        

    
    

