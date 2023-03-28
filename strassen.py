import numpy as np
import time

def standard_multiply(A, B):
    if (A.shape[1] != B.shape[0]):
        print("incompatible dimensions")

    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            for k in range(B.shape[1]):
                C[i][j] += A[i][k]*B[k][j]

    return C
    

def strassen_multiply(A, B, n0):
    m = max(A.shape[0], A.shape[1], B.shape[1])
    m += m % 2  # Round up to next even number
    A_pad = np.zeros((m, m))
    A_pad[:A.shape[0], :A.shape[1]] = A
    # print(A_pad.shape)

    B_pad = np.zeros((m, m))
    # print(B_pad.shape)
    B_pad[:B.shape[0], :B.shape[1]] = B

    if m <= n0 or 0 == n0:
        return standard_multiply(A, B)

    else:
        mid = m // 2
        # mid += mid % 2
        
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

        C = np.zeros((P1.shape[0]*2, P1.shape[1]*2))
        mid = C.shape[0] // 2

        C[:mid, :mid] = P1 + P4 - P5 + P7
        C[:mid, mid:] = P3 + P5
        C[mid:, :mid] = P2 + P4
        C[mid:, mid:] = P1 - P2 + P3 + P6
        return C[:A.shape[1], :B.shape[0]]

n = 99
A = np.random.randint(0, 2, size=(n, n))
B = np.random.randint(0, 2, size=(n, n))

n0 = 23

def switch_test(A, B):
    times = []
    n0_values = [i for i in range(1,32)]
    for n0 in n0_values:
        start_time = time.time()
        strassen_multiply(A, B, n0)
        end_time = time.time()
        times.append(end_time - start_time)
    
    min_time_idx = times.index(min(times))
    optimal_n0 = n0_values[min_time_idx]
    print(optimal_n0)
    return optimal_n0



r_std = standard_multiply(A, B)
print(r_std)

r_stsn = strassen_multiply(A, B, n0)
print(r_stsn)

# switch_test(A,B)

print(np.array_equal(r_std, r_stsn))
