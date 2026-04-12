import numpy as np

#PISTA: es esta la mejor forma de hacer una matmul?
def matmul_biasses_old(A, B, C, bias):
    m, p, n = A.shape[0], A.shape[1], B.shape[1]
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

def matmul_biasses(A, B, C, bias):
    return A @ B + bias

def im2col(input, out_h, out_w, batch_size, k_h, k_w, stride):
    im2col_list = []
    for b in range(batch_size):
        cols = [
            input[ b, :,
                    i * stride : i * stride + k_h,
                    j * stride : j * stride + k_w].reshape(-1)
            for i in range(out_h)
            for j in range(out_w)
        ]

        im2col_list.append(np.array(cols).T)

    return im2col_list