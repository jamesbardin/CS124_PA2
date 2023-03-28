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

def strassen_multiply_pad(A, B, n0):
    n = max(A.shape[0], A.shape[1], B.shape[1])
    m = 1
    while m < n:
        m *= 2

    A_pad = np.zeros((m, m))
    A_pad[:A.shape[0], :A.shape[1]] = A

    B_pad = np.zeros((m, m))
    B_pad[:B.shape[0], :B.shape[1]] = B

    if m <= n0:
        return np.dot(A_pad, B_pad)

    else:
        mid = m // 2

        A11 = A_pad[:mid, :mid]
        A12 = A_pad[:mid, mid:]
        A21 = A_pad[mid:, :mid]
        A22 = A_pad[mid:, mid:]

        B11 = B_pad[:mid, :mid]
        B12 = B_pad[:mid, mid:]
        B21 = B_pad[mid:, :mid]
        B22 = B_pad[mid:, mid:]

        P1 = strassen_multiply(A11 + A22, B11 + B22, n0)
        P2 = strassen_multiply(A21 + A22, B11, n0)
        P3 = strassen_multiply(A11, B12 - B22, n0)
        P4 = strassen_multiply(A22, B21 - B11, n0)
        P5 = strassen_multiply(A11 + A12, B22, n0)
        P6 = strassen_multiply(A21 - A11, B11 + B12, n0)
        P7 = strassen_multiply(A12 - A22, B21 + B22, n0)

        C = np.zeros((m, m))

        C[:mid, :mid] = P1 + P4 - P5 + P7
        C[:mid, mid:] = P3 + P5
        C[mid:, :mid] = P2 + P4
        C[mid:, mid:] = P1 - P2 + P3 + P6

        return C[:A.shape[0], :B.shape[1]]

n = 256
A = np.random.randint(0, 2, size=(n, n))
B = np.random.randint(0, 2, size=(n, n))
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[4, 3], [2, 1]])
n0 = 23



r_std = standard_multiply(A, B)
print(r_std)

r_stsn = strassen_multiply_pad(A, B, n0)
print(r_stsn)

print(np.array_equal(r_std, r_stsn))
