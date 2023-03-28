import numpy as np

def standard_multiply(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            for k in range(B.shape[1]):
                C[i][j] += A[i][k]*B[k][j]

    return C

def strassen_multiply(A, B, n0):
    n = A.shape[0]
    if n <= n0:
        return standard_multiply(A, B)
    else:
        # split up matrices into submatrices
        mid = n // 2
        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]
        B11 = B[:mid, :mid]
        B12 = B[:mid, mid:]
        B21 = B[mid:, :mid]
        B22 = B[mid:, mid:]

        # strassen's
        P1 = strassen_multiply(A11 + A22, B11 + B22, n0)
        P2 = strassen_multiply(A21 + A22, B11, n0)
        P3 = strassen_multiply(A11, B12 - B22, n0)
        P4 = strassen_multiply(A22, B21 - B11, n0)
        P5 = strassen_multiply(A11 + A12, B22, n0)
        P6 = strassen_multiply(A21 - A11, B11 + B12, n0)
        P7 = strassen_multiply(A12 - A22, B21 + B22, n0)

        # result matrix
        C = np.zeros((n, n))
        
        # fill result in w/ combined submatrices
        C[:mid, :mid] = P1 + P4 - P5 + P7
        C[:mid, mid:] = P3 + P5
        C[mid:, :mid] = P2 + P4
        C[mid:, mid:] = P1 - P2 + P3 + P6
        
        return C
    

A = np.random.randint(0, 2, size=(128, 128))
B = np.random.randint(0, 2, size=(128, 128))
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[4, 3], [2, 1]])
n0 = 23



r_std = standard_multiply(A, B)
print(r_std)

r_stsn = strassen_multiply(A, B, n0)
print(r_stsn)

print(np.array_equal(r_std, r_stsn))
