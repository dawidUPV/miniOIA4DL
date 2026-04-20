from modules.layer import Layer
from modules.utils import im2col
try:
    from cython_modules.im2col import im2col_forward_cython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward_old(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        output = np.zeros((B, C, out_h, out_w),dtype=input.dtype)

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_val = window[max_idx]

                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        return output

    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        # --- INICIO BLOQUE GENERADO CON IA ---
        # Se ha usado IA para utilizar im2col con max pooling y asi evitar los bucles anidados

        # B matrices of shape [C * KH * KW, out_h * out_w]
        im2col_list = im2col(input, out_h, out_w, B, KH, KW, self.stride)
        if CYTHON_AVAILABLE:
            im2col_array = im2col_forward_cython(input.astype(np.float32), out_h, out_w, B, KH, KW, self.stride)
            im2col_list = [im2col_array[b] for b in range(B)]
        else:
            im2col_list = im2col(input, out_h, out_w, B, KH, KW, self.stride)
        
        # A single 3D array [B, C * KH * KW, out_h * out_w]
        col_stacked = np.array(im2col_list)
        
        # Reshape to [B, C, KH * KW, out_h * out_w]
        col_reshaped = col_stacked.reshape(B, C, KH * KW, out_h * out_w)

        # Perform the max operation along the window dimension (axis 2)
        output = np.max(col_reshaped, axis=2)

        # --- FIN BLOQUE GENERADO CON IA ---
        
        # El siguiente bloque es para guardar los indices de los maximos, para el backward
        # aunque no se utiliza en este proyecto
        
        # --- INICIO BLOQUE GENERADO CON IA ---
        # Get indices of max values along window dimension
        max_indices_linear = np.argmax(col_reshaped, axis=2)  # shape: (B, C, out_h, out_w)
        
        # Convert linear indices to (h_offset, w_offset) within each patch
        h_offsets, w_offsets = np.unravel_index(max_indices_linear.flatten(), (KH, KW))
        h_offsets = h_offsets.reshape(B, C, out_h, out_w)
        w_offsets = w_offsets.reshape(B, C, out_h, out_w)
        
        # Convert to absolute coordinates by adding stride offsets
        h_starts = np.arange(out_h).reshape(1, 1, out_h, 1) * SH
        w_starts = np.arange(out_w).reshape(1, 1, 1, out_w) * SW
        
        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        self.max_indices[:, :, :, :, 0] = h_starts + h_offsets
        self.max_indices[:, :, :, :, 1] = w_starts + w_offsets
        # --- FIN BLOQUE GENERADO CON IA ---
        
        # Reshape back to the standard [B, C, out_h, out_w]
        output = output.reshape(B, C, out_h, out_w)
        
        return output

    def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input