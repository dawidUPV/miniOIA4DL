import numpy as np
cimport numpy as np

# --- INICIO BLOQUE GENERADO CON IA ---
ctypedef float DTYPE_t

def im2col_forward_cython(float[:, :, :, :] input, 
                          int out_h, int out_w, 
                          int batch_size, int k_h, int k_w, int stride):
    
    cdef int in_channels = input.shape[1]
    cdef int row_size = in_channels * k_h * k_w
    cdef int col_size = out_h * out_w
    
    cdef float[:, :, :] output = np.zeros((batch_size, row_size, col_size), dtype=np.float32)
    
    cdef int b, c, i, j, ii, jj
    cdef int out_col, channel_offset, row_offset
    
    for b in range(batch_size):
        for c in range(in_channels):
            channel_offset = c * k_h * k_w
            for ii in range(k_h):
                row_offset = channel_offset + ii * k_w
                for jj in range(k_w):
                    for i in range(out_h):
                        for j in range(out_w):
                            out_col = i * out_w + j
                            output[b, row_offset + jj, out_col] = \
                                input[b, c, i * stride + ii, j * stride + jj]
                                
    return np.asarray(output)
# --- FIN BLOQUE GENERADO CON IA ---
