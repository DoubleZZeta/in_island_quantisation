# A torch based simulator for the PE GEMV operation, used for testing and debugging the Triton implementation.
import torch
import triton
import triton.language as tl
from . import quantization as q

def quantized_pe_gemv(x, W, pe_rows=3, pe_cols=3, partial_bits=8):
    K, N = W.shape
    k_per_pe = triton.cdiv(K, pe_rows)
    n_per_pe = triton.cdiv(N, pe_cols)

    y = torch.zeros((N,), device=x.device, dtype=torch.float32)
    partials = []
    scales = []

    for pe_row in range(pe_rows):
        row_start = pe_row * k_per_pe
        row_end = min(row_start + k_per_pe, K)
        partial_row = []
        scale_row = []

        for pe_col in range(pe_cols):
            col_start = pe_col * n_per_pe
            col_end = min(col_start + n_per_pe, N)

            partial_fp = x[row_start:row_end].float() @ W[row_start:row_end, col_start:col_end].float()
            partial_q, scale = q.quantize_symmetric(partial_fp, num_bits=partial_bits)
            y[col_start:col_end] += q.dequantize_symmetric(partial_q, scale)

            partial_row.append(partial_q)
            scale_row.append(scale)

        partials.append(partial_row)
        scales.append(torch.stack(scale_row))

    return y, partials, torch.stack(scales)
