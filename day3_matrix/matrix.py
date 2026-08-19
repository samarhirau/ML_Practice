import numpy as np


A=np.eye(3)          # Identity matrix
B=np.zeros((2, 3))   # All zeros
C=np.ones((2, 3))    # All ones
D=np.random.rand(2, 3)  # Random values

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)